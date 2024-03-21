import torch
import torch.nn as nn
import math
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


class ComplexNorm(nn.Module):
    def __init__(self, type):
        super(ComplexNorm, self).__init__()
        self.type = type 
    
    def forward(self, x):
        # x.shape: b, c, h, w
        if self.type == 'spatial':
            # https://jermwatt.github.io/machine_learning_refined/notes/16_Linear_algebra/16_5_Norms.html#:~:text=For%20example%2C%20the%20Frobenius%20norm,%2B52%3D%E2%88%9A30.
            # Frobenius norm is the intuitive extension of the ℓ2 norm for vectors to matrices
            norm = torch.linalg.matrix_norm(x, dim=(-2, -1))
            norm = norm.unsqueeze(-1).unsqueeze(-1)
            out = torch.div(x, norm)
        elif self.type == 'channel':
            # take the square root of the sum of the squared vector values
            norm = torch.linalg.vector_norm(x, dim=1)
            norm = norm.unsqueeze(1)
            out = torch.div(x, norm)
        elif self.type == 'last_2nd_dim':
            norm = torch.linalg.vector_norm(x, dim=-2)
            norm = norm.unsqueeze(-2)
            out = torch.div(x, norm)
        else: 
            norm = torch.linalg.vector_norm(x, dim=-1)
            norm = norm.unsqueeze(-1)
            out = torch.div(x, norm)
        return out
        
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
        
        self.de_block_1 = ChanBlock(dim, head_num, 1, False)
        self.de_block_2 = ChanBlock(dim, head_num, 1, False)
        self.de_block_3 = ChanBlock(dim, head_num, 1, False)
        self.de_block_4 = ChanBlock(dim, head_num, 1, False)
        self.de_block_5 = ChanBlock(dim, head_num, 1, False)
        self.de_block_6 = ChanBlock(dim, head_num, 1, False)


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

        hx = self.de_block_1(hx)
        hx = self.de_block_2(hx)
        hx = self.de_block_3(hx)
        hx = self.de_block_4(hx)
        hx = self.de_block_5(hx)
        hx = self.de_block_6(hx)

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


class SpatialBlock(nn.Module):
    def __init__(self, dim, num_heads, win_size, ffn_expansion_factor, bias):
        super(SpatialBlock, self).__init__()

        self.heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1, 1))
        
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.complex_norm = ComplexNorm(type='last_dim')
        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.patch_size = win_size

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

    def attn(self, x):
        B, C, H, W= x.shape
        
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b (head c) (h patch1) (w patch2) -> b head (h w) c (patch1 patch2)', head=self.heads, patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b (head c) (h patch1) (w patch2) -> b head (h w) c (patch1 patch2)', head=self.heads, patch1=self.patch_size,
                            patch2=self.patch_size)
        v_patch = rearrange(v, 'b (head c) (h patch1) (w patch2) -> b head (h w) c (patch1 patch2)', head=self.heads, patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft(q_patch.float(), dim=-1)
        k_fft = torch.fft.rfft(k_patch.float(), dim=-1)
        v_fft = torch.fft.rfft(v_patch.float(), dim=-1)
        
        attn = (q_fft.transpose(-2, -1) @ k_fft) * self.temperature  # b head (h w) (patch1 patch2) (patch1 patch2)
        
        attn = self.complex_norm(attn)  # b head (h w) (patch1 patch2) (patch1 patch2)
        
        out = attn @ (v_fft.transpose(-2, -1))  # b head (h w) (patch1 patch2) c
        out = out.transpose(-2, -1)  # b head (h w) c (patch1 patch2)
        out = torch.fft.irfft(out, dim=-1)  # b head (h w) c (patch1 patch2)
        
        out = rearrange(out, 'b head (h w) c (patch1 patch2) -> b (head c) (h patch1) (w patch2)', head=self.heads, 
                        h=H//self.patch_size, w=W//self.patch_size, patch1=self.patch_size, patch2=self.patch_size)
        
        out = self.project_out(out)

        return out
"""
import time
start_time = time.time()
inp = torch.randn(1, 32, 64, 64).cuda()
model = SpatialBlock(dim=32, num_heads=2, win_size=8, ffn_expansion_factor=1, bias=False).cuda()
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""

class ChanBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias):
        super(ChanBlock, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


    def attn(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
"""
import time
start_time = time.time()
inp = torch.randn(1, 32, 64, 64).cuda()
model = ChanBlock(dim=32, num_heads=2, bias=False).cuda()
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()

        self.encoder = Embeddings()
        head_num = 5
        dim = 320
        #dim = 320
        self.Trans_block_1 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_2 = ChanBlock(dim, head_num, 1, False)  # dim, num_heads, ffn_expansion_factor, bias
        self.Trans_block_3 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_4 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_5 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_6 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_7 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_8 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_9 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_10 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_11 = ChanBlock(dim, head_num, 1, False)
        self.Trans_block_12 = ChanBlock(dim, head_num, 1, False)
        self.decoder = Embeddings_output()


    def forward(self, x):

        hx, residual_1, residual_2 = self.encoder(x)
        hx = self.Trans_block_1(hx)
        hx = self.Trans_block_2(hx)
        hx = self.Trans_block_3(hx)
        hx = self.Trans_block_4(hx)
        hx = self.Trans_block_5(hx)
        hx = self.Trans_block_6(hx)
        hx = self.Trans_block_7(hx)
        hx = self.Trans_block_8(hx)
        hx = self.Trans_block_9(hx)
        hx = self.Trans_block_10(hx)
        hx = self.Trans_block_11(hx)
        hx = self.Trans_block_12(hx)
        hx = self.decoder(hx, residual_1, residual_2)

        return hx + x
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda()
model = Freqformer_V2().cuda()
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""