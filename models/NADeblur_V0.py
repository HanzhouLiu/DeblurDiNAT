import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numbers
#from models.torch_wavelets import DWT_2D, IDWT_2D
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D

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

# RFM (Residual fusion module, multi-scale)
class RFM(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, kernel, dilation, ffn_expansion_factor, bias):
        super(RFM, self).__init__()
        self.ResEmbedding = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                                          nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.FusionBlock = TransBlock(out_dim, num_heads, kernel, 
                                      dilation, ffn_expansion_factor, bias)
        
    def forward(self, x1, x2, x3):
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.ResEmbedding(x)
        x = self.FusionBlock(x)
        return x

# gated fusion module 
class Fusion(nn.Module):
    def __init__(self, n_feat, bias):
        super(Fusion, self).__init__()

        self.dwconv = nn.Conv2d(n_feat*2, n_feat*2, kernel_size=3, stride=1, padding=1, groups=n_feat*2, bias=bias)
        self.out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def forward(self, en, de):
        x = self.dwconv(torch.cat((en, de), dim=1))
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2 
        x = self.out(x)
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

class TransBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel, dilation, ffn_expansion_factor, bias):
        super(TransBlock, self).__init__()

        self.heads = num_heads
        self.kernel = kernel
        self.dilation = dilation
        self.na2d = NeighborhoodAttention2D(dim=dim, kernel_size=kernel, 
                                            dilation=dilation, num_heads=num_heads)
        self.fusion = Fusion(dim, bias)
        
        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        y = self.attn(self.norm1(x))
        x = self.fusion(x, y)
        y = self.ffn(self.norm2(x))
        x = self.fusion(x, y)

        return x

    def attn(self, x):
        # b, c, h, w -> b, h, w, c
        x = x.permute(0, 2, 3, 1)
        x = self.na2d(x)
        x = x.permute(0, 3, 1, 2)
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
            TransBlock(dim*2**2, heads[2], 7, 4, 1, bias=bias) for i in 
            range(num_blocks[2])
        ])
        
        self.up3_2 = nn.Sequential(
            nn.ConvTranspose2d(dim*2**2, dim*2, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level2 = Fusion(n_feat=dim*2, bias=bias)

        self.de_trans_level2 = nn.Sequential(*[
            TransBlock(dim*2, heads[1], 7, 8, 1, bias=bias) for i in 
            range(num_blocks[1])
        ])

        self.up2_1 = nn.Sequential(
            nn.ConvTranspose2d(dim*2, dim, kernel_size=4, stride=2, padding=1, bias=bias),
            self.activation,
        )

        self.fusion_level1 = Fusion(n_feat=dim, bias=bias)

        self.de_trans_level1 = nn.Sequential(*[
            TransBlock(dim, heads[0], 7, 16, 1, bias=bias) for i in 
            range(num_blocks[0])
        ])
        
        self.refinement = nn.Sequential(*[
            TransBlock(dim, heads[0], 7, 16, 1, bias=bias) for i in 
            range(num_refinement_blocks)
        ])
        self.output = nn.Sequential(
            nn.Conv2d(dim, 3, kernel_size=3, padding=1, bias=bias),
            self.activation
        )

    def forward(self, x, residual_1, residual_2):
        hx = self.de_trans_level3(x)
        hx = self.up3_2(hx)
        hx = self.fusion_level2(hx, residual_2)
        hx = self.de_trans_level2(hx)
        hx = self.up2_1(hx)
        hx = self.fusion_level1(hx, residual_1)
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
class NADeblur_V0(nn.Module):
    def __init__(self,
                 dim = 64, 
                 num_blocks = [4,6,8],
                 num_refinement_blocks = 4,
                 num_heads = [2,4,8], 
                 kernel = 7, 
                 dilation = 3, 
                 ffn_expansion_factor = 1, 
                 bias = False):
        super(NADeblur_V0, self).__init__()

        #self.dwt = DWT_2D(wave='haar')
        #self.idwt = IDWT_2D(wave='haar')
        
        self.encoder = Embeddings(dim)
        #dim = 320
        self.RFM1 = RFM(dim*7, dim*1, num_heads[0], 7, 16, ffn_expansion_factor, bias)
        self.RFM2 = RFM(dim*7, dim*2, num_heads[1], 7, 8, ffn_expansion_factor, bias)
    
        self.decoder = Embeddings_output(dim, num_blocks, num_refinement_blocks, 
                                         num_heads, bias)


    def forward(self, x):
        
        hx, res1, res2 = self.encoder(x)
        
        # level1: res1, level2: res2, level3: hx)
        
        res2_1 = F.interpolate(res2, scale_factor=2)
        res1_2 = F.interpolate(res1, scale_factor=0.5)
        hx_2   = F.interpolate(hx, scale_factor=2)
        hx_1   = F.interpolate(hx_2, scale_factor=2)
        
        res1 = self.RFM1(res1, res2_1, hx_1)
        res2 = self.RFM2(res1_2, res2, hx_2)

        hx = self.decoder(hx, res1, res2)

        return hx + x
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda()#.to(dtype=torch.float16)
model = NADeblur_V1().cuda()#.to(dtype=torch.float16)
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""
