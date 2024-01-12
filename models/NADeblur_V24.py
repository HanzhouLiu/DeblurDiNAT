import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numbers
#from models.torch_wavelets import DWT_2D, IDWT_2D
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x

# Gated-Dconv Fusion Block (GDFB)
class GatedFusion(nn.Module):
    def __init__(self, in_dim, out_dim, ffn_expansion_factor, bias):
        super(GatedFusion, self).__init__()
        self.project_in = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                                          nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.norm = LayerNorm(out_dim, LayerNorm_type = 'BiasFree')
        self.ffn = GDFN(out_dim, ffn_expansion_factor, bias)
        
    def forward(self, x):
        x = self.project_in(x)
        x = x + self.ffn(self.norm(x))
        return x
        
    ##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class GDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Attention(nn.Module):
    def __init__(self, head_num):
        super(Attention, self).__init__()
        self.num_attention_heads = head_num
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        B, N, C = x.size()
        attention_head_size = int(C / self.num_attention_heads)
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, query_layer, key_layer, value_layer):
        B, N, C = query_layer.size()
        query_layer = self.transpose_for_scores(query_layer)
        key_layer = self.transpose_for_scores(key_layer)
        value_layer = self.transpose_for_scores(value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        _, _, _, d = query_layer.size()
        attention_scores = attention_scores / math.sqrt(d)
        attention_probs = self.softmax(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (C,)
        attention_out = context_layer.view(*new_context_layer_shape)

        return attention_out


    ##########################################################################
## Dual Branch Gated-Dconv Feed-Forward Network (DBGDFN)
class DBGDFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(DBGDFN, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv_br1 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias, dilation=1)
        
        #self.dwconv_br2 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=2, groups=hidden_features*2, bias=bias, dilation=2)
        
        #self.dwconv_br3 = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=3, groups=hidden_features*2, bias=bias, dilation=3)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv_br1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * x2
        x = self.project_out(x)
        return x


class Intra_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Intra_SA, self).__init__()
        self.hidden_size = dim // 2
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_input = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.qkv_local_h = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_h
        self.qkv_local_v = nn.Linear(self.hidden_size, self.hidden_size * 3)  # qkv_v
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.ffn_norm = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = DBGDFN(dim, ffn_expansion_factor=1, bias=False)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)
    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = (x_input[0]).permute(0, 2, 3, 1).contiguous()
        feature_h = feature_h.view(B * H, W, C//2)
        feature_v = (x_input[1]).permute(0, 3, 2, 1).contiguous()
        feature_v = feature_v.view(B * W, H, C//2)
        qkv_h = torch.chunk(self.qkv_local_h(feature_h), 3, dim=2)
        qkv_v = torch.chunk(self.qkv_local_v(feature_v), 3, dim=2)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]
        q_v, k_v, v_v = qkv_v[0], qkv_v[1], qkv_v[2]

        if H == W:
            query = torch.cat((q_h, q_v), dim=0)
            key = torch.cat((k_h, k_v), dim=0)
            value = torch.cat((v_h, v_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(q_h, k_h, v_h)
            attention_output_v = self.attn(q_v, k_v, v_v)
            attention_output_h = attention_output_h.view(B, H, W, C//2).permute(0, 3, 1, 2).contiguous()
            attention_output_v = attention_output_v.view(B, W, H, C//2).permute(0, 3, 2, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        #x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        #x = x.permute(0, 2, 1).contiguous()
        #x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


class Inter_SA(nn.Module):
    def __init__(self,dim, head_num):
        super(Inter_SA, self).__init__()
        self.hidden_size = dim
        self.head_num = head_num
        self.attention_norm = nn.LayerNorm(self.hidden_size)
        self.conv_input = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.conv_h = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_h
        self.conv_v = nn.Conv2d(self.hidden_size//2, 3 * (self.hidden_size//2), kernel_size=1, padding=0)  # qkv_v
        self.ffn_norm = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = DBGDFN(self.hidden_size, ffn_expansion_factor=1, bias=False)
        self.fuse_out = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=1, padding=0)
        self.attn = Attention(head_num=self.head_num)
        self.PEG = PEG(dim)

    def forward(self, x):
        h = x
        B, C, H, W = x.size()

        x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()
        x = x.view(B, C, H, W)

        x_input = torch.chunk(self.conv_input(x), 2, dim=1)
        feature_h = torch.chunk(self.conv_h(x_input[0]), 3, dim=1)
        feature_v = torch.chunk(self.conv_v(x_input[1]), 3, dim=1)
        query_h, key_h, value_h = feature_h[0], feature_h[1], feature_h[2]
        query_v, key_v, value_v = feature_v[0], feature_v[1], feature_v[2]

        horizontal_groups = torch.cat((query_h, key_h, value_h), dim=0)
        horizontal_groups = horizontal_groups.permute(0, 2, 1, 3).contiguous()  # b h c w
        horizontal_groups = horizontal_groups.view(3*B, H, -1)  # b h (c w)
        horizontal_groups = torch.chunk(horizontal_groups, 3, dim=0)
        query_h, key_h, value_h = horizontal_groups[0], horizontal_groups[1], horizontal_groups[2]

        vertical_groups = torch.cat((query_v, key_v, value_v), dim=0)
        vertical_groups = vertical_groups.permute(0, 3, 1, 2).contiguous()  # b w c h
        vertical_groups = vertical_groups.view(3*B, W, -1)  # b w (c h)
        vertical_groups = torch.chunk(vertical_groups, 3, dim=0)
        query_v, key_v, value_v = vertical_groups[0], vertical_groups[1], vertical_groups[2]


        if H == W:
            query = torch.cat((query_h, query_v), dim=0)
            key = torch.cat((key_h, key_v), dim=0)
            value = torch.cat((value_h, value_v), dim=0)
            attention_output = self.attn(query, key, value)
            attention_output = torch.chunk(attention_output, 2, dim=0)
            attention_output_h = attention_output[0]
            attention_output_v = attention_output[1]
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)
            attention_output_v = self.attn(query_v, key_v, value_v)
            attention_output_h = attention_output_h.view(B, H, C//2, W).permute(0, 2, 1, 3).contiguous()
            attention_output_v = attention_output_v.view(B, W, C//2, H).permute(0, 2, 3, 1).contiguous()
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))

        x = attn_out + h
        #x = x.view(B, C, H*W).permute(0, 2, 1).contiguous()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        #x = x.permute(0, 2, 1).contiguous()
        #x = x.view(B, C, H, W)

        x = self.PEG(x)

        return x


class Embeddings(nn.Module):
    def __init__(self, dim):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim, dim, kernel_size=3, padding=1))
        #self.en_layer1_4 = nn.Sequential(
        #    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
        #    self.activation,
        #    nn.Conv2d(dim, dim, kernel_size=3, padding=1))


        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1))
        #self.en_layer2_4 = nn.Sequential(
        #    nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1),
        #    self.activation,
        #    nn.Conv2d(dim*2, dim*2, kernel_size=3, padding=1))


        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(dim*2, dim*2**2, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer3_2 = nn.Sequential(
            nn.Conv2d(dim*2**2, dim*2**2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim*2**2, dim*2**2, kernel_size=3, padding=1))
        self.en_layer3_3 = nn.Sequential(
            nn.Conv2d(dim*2**2, dim*2**2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim*2**2, dim*2**2, kernel_size=3, padding=1))


    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        #hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx
        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        #hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx
        hx = self.en_layer3_1(hx)
        hx = self.activation(self.en_layer3_2(hx) + hx)
        hx = self.activation(self.en_layer3_3(hx) + hx)

        return hx, residual_1, residual_2

class Embeddings_output(nn.Module):
    def __init__(self, dim, num_blocks, num_refinement_blocks, heads, bias):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)
        
        self.de_trans_level3 = nn.Sequential(*[
            item for sublist in 
            [[Intra_SA(dim*2**2, heads[2]),
             Inter_SA(dim*2**2, heads[2])] for i in range(num_blocks[2]//2)] for item in sublist
        ])
        
        self.up3_2 = nn.Sequential(
            nn.ConvTranspose2d(dim*2**2, dim*2, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level2 = GatedFusion(dim*4, dim*2, 1, bias)

        self.de_trans_level2 = nn.Sequential(*[
            item for sublist in 
            [[Intra_SA(dim*2, heads[1]),
             Inter_SA(dim*2, heads[1])] for i in range(num_blocks[1]//2)] for item in sublist
        ])

        self.up2_1 = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level1 = GatedFusion(dim*2, dim*1, 1, bias)


        self.de_trans_level1 = nn.Sequential(*[
            item for sublist in 
            [[Intra_SA(dim, heads[0]),
             Inter_SA(dim, heads[0])] for i in range(num_blocks[0]//2)] for item in sublist
        ])
        
        self.refinement = nn.Sequential(*[
            item for sublist in 
            [[Intra_SA(dim, heads[0]),
             Inter_SA(dim, heads[0])] for i in range(num_blocks[0]//2)] for item in sublist
        ])
        self.output = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, padding=1, bias=bias),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_trans_level3(x)
        hx = self.up3_2(hx)
        hx = self.fusion_level2(torch.cat((hx, residual_2), dim=1))
        hx = self.de_trans_level2(hx)
        hx = self.up2_1(hx)
        hx = self.fusion_level1(torch.cat((hx, residual_1), dim=1))
        hx = self.de_trans_level1(hx)
        hx = self.refinement(hx)
        hx = self.output(hx)
        return hx

"""
import time
start_time = time.time()
inp = torch.randn(1, 256, 32, 32).cuda()#.to(dtype=torch.float16)
res1 = torch.randn(1, 64, 128, 128).cuda()
res2 = torch.randn(1, 128, 64, 64).cuda()
model = Embeddings_output(dim=64, num_blocks = [4,6,8], num_refinement_blocks=4, 
                                         heads = [2,4,8], bias=False).cuda()#.to(dtype=torch.float16)
out = model(inp, res1, res2)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""
class NADeblur_V24(nn.Module):
    def __init__(self,
                 dim = 64, 
                 num_blocks = [4,6,8],
                 num_refinement_blocks = 4,
                 num_heads = [2,4,8], 
                 kernel = 7, 
                 dilation = 3, 
                 ffn_expansion_factor = 1, 
                 bias = False):
        super(NADeblur_V24, self).__init__()
        
        self.encoder = Embeddings(dim)
        #dim = 320
        self.RFM1 = GatedFusion(dim*7, dim*1, ffn_expansion_factor, bias)
        self.RFM2 = GatedFusion(dim*7, dim*2, ffn_expansion_factor, bias)
    
        self.decoder = Embeddings_output(dim, num_blocks, num_refinement_blocks, 
                                         num_heads, bias)


    def forward(self, x):
        
        hx, res1, res2 = self.encoder(x)
        
        # level1: res1, level2: res2, level3: hx)
        
        res2_1 = F.interpolate(res2, scale_factor=2)
        res1_2 = F.interpolate(res1, scale_factor=0.5)
        hx_2   = F.interpolate(hx, scale_factor=2)
        hx_1   = F.interpolate(hx_2, scale_factor=2)
        
        res1 = self.RFM1(torch.cat((res1, res2_1, hx_1), dim=1))
        res2 = self.RFM2(torch.cat((res1_2, res2, hx_2), dim=1))

        hx = self.decoder(hx, res1, res2)

        return hx + x
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda()#.to(dtype=torch.float16)
model = NADeblur_V24().cuda()#.to(dtype=torch.float16)
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""
