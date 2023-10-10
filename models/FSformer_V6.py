import torch
import torch.nn as nn
import math
from torch_wavelets import DWT_2D, IDWT_2D
import torch.nn.functional as F
import numbers

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


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        
    def forward(self, x):
        out = self.deconv(x)
        return out


class Embeddings(nn.Module):
    def __init__(self):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1))


        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            self.activation,
        )


    def forward(self, x):

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        residual_1 = hx
        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        residual_2 = hx
        hx = self.en_layer3_1(hx)

        return hx, residual_1, residual_2


class Embeddings_output(nn.Module):
    def __init__(self):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(320, 192, kernel_size=4, stride=2, padding=1),
            self.activation,
        )
        head_num = 3
        dim = 192
        #dim = 192

        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(192+128, 192, kernel_size=1, padding=0),
            self.activation,
        )
        
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.de_block_1 = LoBlock(dim, head_num, 4, 1, False)
        self.de_block_2 = HiBlock(dim, head_num, 8, 1, False)
        self.de_block_3 = LoBlock(dim, head_num, 4, 1, False)
        self.de_block_4 = HiBlock(dim, head_num, 8, 1, False)
        self.de_block_5 = LoBlock(dim, head_num, 4, 1, False)
        self.de_block_6 = HiBlock(dim, head_num, 8, 1, False)


        self.de_layer2_1 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
            self.activation,
        )

        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, padding=0),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):


        hx = self.de_layer3_1(x)

        hx = self.de_layer2_2(torch.cat((hx, residual_2), dim = 1))
        
        ll, lh, hl, hh = self.dwt(hx).chunk(4, dim=1)
        ll = self.de_block_1(ll)
        hx = self.de_block_2(hx)
        ll = self.de_block_3(ll)
        hx = self.de_block_4(hx)
        ll = self.de_block_5(ll)
        hx = self.de_block_6(hx)
        ll_, lh_, hl_, hh_ = self.dwt(hx).chunk(4, dim=1)
        hx = self.idwt(torch.cat((ll, lh_, hl_, hh_), dim=1))

        hx = self.de_layer2_1(hx)
        hx = self.activation(self.de_layer1_3(torch.cat((hx, residual_1), dim = 1)) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.de_layer1_1(hx)

        return hx

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


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4*hidden_size)
        self.fc2 = nn.Linear(4*hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


# CPE (Conditional Positional Embedding)
class PEG(nn.Module):
    def __init__(self, hidden_size):
        super(PEG, self).__init__()
        self.PEG = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=hidden_size)

    def forward(self, x):
        x = self.PEG(x) + x
        return x

    
    ##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

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


class HiBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size, ffn_expansion_factor, bias):
        super(HiBlock, self).__init__()

        self.heads = num_heads
        self.ws = win_size
        self.scale = num_heads ** -0.5
        #self.h_qkv = nn.Linear(self.h_dim, self.h_dim*3, bias=bias)
        self.h_qkv = nn.Sequential(
            nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias),
            nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        )
        
        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    
    def attn(self, x):
        """
        Similar to HiLo Attention

        Paper: Fast Vision Transformers with HiLo Attention
        Link: https://arxiv.org/abs/2205.13213
        """
        
        B, C, H, W= x.shape

        h_qkv = self.h_qkv(x)
        h_q, h_k, h_v = h_qkv.chunk(3, dim=1)

        h_q = rearrange(h_q, 'b (head c) (h ws1) (w ws2) -> b head (h w) (ws1 ws2) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)
        h_k = rearrange(h_k, 'b (head c) (h ws1) (w ws2) -> b head (h w) (ws1 ws2) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)
        h_v = rearrange(h_v, 'b (head c) (h ws1) (w ws2) -> b head (h w) (ws1 ws2) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)
        # B, n_head, hw, ws*ws, head_dim

        h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale  # B, n_head, hw, ws*ws, ws*ws
        h_attn = h_attn.softmax(dim=-1)

        h_attn = (h_attn @ h_v)

        x = rearrange(h_attn, 'b head (h w) (ws1 ws2) c -> b (head c) (h ws1) (w ws2)', head=self.heads, 
                      ws1=self.ws, ws2=self.ws, h=H//self.ws, w=W//self.ws)
        
        return x


class LoBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size, ffn_expansion_factor, bias):
        super(LoBlock, self).__init__()

        self.heads = num_heads
        self.ws = win_size
        self.scale = num_heads ** -0.5
        #self.h_qkv = nn.Linear(self.h_dim, self.h_dim*3, bias=bias)
        self.l_qkv = nn.Sequential(
            nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias),
            nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        )
        
        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

    def attn(self, x):
        """
        Similar to HiLo Attention

        Paper: Fast Vision Transformers with HiLo Attention
        Link: https://arxiv.org/abs/2205.13213
        """
        
        B, C, H, W= x.shape

        l_qkv = self.l_qkv(x)
        l_q, l_k, l_v = l_qkv.chunk(3, dim=1)

        l_q = rearrange(l_q, 'b (head c) (h ws1) (w ws2) -> b head (ws1 ws2) (h w) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)
        l_k = rearrange(l_k, 'b (head c) (h ws1) (w ws2) -> b head (ws1 ws2) (h w) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)
        l_v = rearrange(l_v, 'b (head c) (h ws1) (w ws2) -> b head (ws1 ws2) (h w) c', head=self.heads, 
                        ws1=self.ws, ws2=self.ws)


        l_attn = (l_q @ l_k.transpose(-2, -1)) * self.scale  # B, n_head, hw, h*w, h*w
        l_attn = l_attn.softmax(dim=-1)

        l_attn = (l_attn @ l_v)

        x = rearrange(l_attn, 'b head (ws1 ws2) (h w) c -> b (head c) (h ws1) (w ws2)', head=self.heads, 
                        ws1=self.ws, ws2=self.ws, h=H//self.ws, w=W//self.ws)
        
        return x


class FSformer_V6(nn.Module):
    def __init__(self):
        super(FSformer_V6, self).__init__()

        self.encoder = Embeddings()
        head_num = 5
        dim = 320
        #dim = 320
        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.Trans_block_1 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_2 = HiBlock(dim, head_num, 8, 1, False)
        self.Trans_block_3 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_4 = HiBlock(dim, head_num, 8, 1, False)
        self.Trans_block_5 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_6 = HiBlock(dim, head_num, 8, 1, False)
        self.Trans_block_7 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_8 = HiBlock(dim, head_num, 8, 1, False)
        self.Trans_block_9 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_10 = HiBlock(dim, head_num, 8, 1, False)
        self.Trans_block_11 = LoBlock(dim, head_num, 4, 1, False)
        self.Trans_block_12 = HiBlock(dim, head_num, 8, 1, False)
        self.decoder = Embeddings_output()


    def forward(self, x):

        hx, residual_1, residual_2 = self.encoder(x)
        ll, lh, hl, hh = self.dwt(hx).chunk(4, dim=1)
        ll = self.Trans_block_1(ll)
        hx = self.Trans_block_2(hx)
        ll = self.Trans_block_3(ll)
        hx = self.Trans_block_4(hx)
        ll = self.Trans_block_5(ll)
        hx = self.Trans_block_6(hx)
        ll = self.Trans_block_7(ll)
        hx = self.Trans_block_8(hx)
        ll = self.Trans_block_9(ll)
        hx = self.Trans_block_10(hx)
        ll = self.Trans_block_11(ll)
        hx = self.Trans_block_12(hx)
        ll_, lh_, hl_, hh_ = self.dwt(hx).chunk(4, dim=1)
        hx = self.idwt(torch.cat((ll, lh_, hl_, hh_), dim=1))
        hx = self.decoder(hx, residual_1, residual_2)

        return hx + x
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda().to(dtype=torch.float16)
model = FSformer_V6().cuda().to(dtype=torch.float16)
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""