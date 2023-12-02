import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numbers
#from models.torch_wavelets import DWT_2D, IDWT_2D
from natten import NeighborhoodAttention1D, NeighborhoodAttention2D
from models.base_networks import Encoder_MDCBlock1, Decoder_MDCBlock1
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
    def __init__(self, dim):
        super(RFM, self).__init__()
        self.conv1_1 = nn.Conv2d(dim, 256, kernel_size=1)
        self.conv2_1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv3_1 = nn.Conv2d(dim*2, 256, kernel_size=1)
        self.conv1_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv2_2 = nn.Conv2d(dim*4, 256, kernel_size=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x1, x2, x3):
        x1n = self.conv1_1(x1)
        x2n = self.conv2_1(x2)
        x3n = self.conv3_1(x3)
        g1 = self.conv1_2(x1n)
        g2 = self.conv2_2(x2n)
        g3 = self.conv3_2(x3n)
        g1 = self.sigmoid(g1)
        g2 = self.sigmoid(g2)
        g3 = self.sigmoid(g3)
        x = (1+g1)*x1n + (1-g1)*(g2*x2n + g3*x3n)
        return x

# normal fusion module 
class Fusion(nn.Module):
    def __init__(self, n_feat, bias):
        super(Fusion, self).__init__()
        
        self.conv = nn.Conv2d(n_feat*2, n_feat, kernel_size=1, stride=1, bias=bias)
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, en, de):
        x = self.conv(torch.cat((en, de), dim=1))
        x = self.activation(x)
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
        
        self.norm1 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.norm2 = LayerNorm(dim, LayerNorm_type = 'BiasFree')
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

    def attn(self, x):
        # b, c, h, w -> b, h, w, c
        x = x.permute(0, 2, 3, 1)
        x = self.na2d(x)
        x = x.permute(0, 3, 1, 2)
        return x

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

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
        
        self.fusion1 = Encoder_MDCBlock1(dim, 2, mode='iter2')
        self.fusion2 = Encoder_MDCBlock1(dim*2, 3, mode='iter2')
        
        self.conv1 = RDB(64, 4, 64)
        self.conv2 = RDB(128, 4, 128)

    def forward(self, x):

        hx1 = self.en_layer1_1(x)
        hx1_1, hx1_2 = hx1.split([(hx1.size()[1] // 2), (hx1.size()[1] // 2)], dim=1)
        feat_mem = [hx1_1]
        hx1_ = self.activation(self.en_layer1_2(hx1) + hx1)
        hx1_ = self.activation(self.en_layer1_3(hx1_) + hx1_)
        hx1 = hx1_ + hx1
        #hx = self.activation(self.en_layer1_4(hx) + hx)
        hx2 = self.en_layer2_1(hx1)
        hx2_1, hx2_2 = hx2.split([(hx2.size()[1] // 2), (hx2.size()[1] // 2)], dim=1)
        hx2_1 = self.fusion1(hx2_1, feat_mem)
        hx2_2 = self.conv1(hx2_2)
        feat_mem.append(hx2_1)
        hx2 = torch.cat((hx2_1, hx2_2), dim=1)
        hx2_ = self.activation(self.en_layer2_2(hx2) + hx2)
        hx2_ = self.activation(self.en_layer2_3(hx2_) + hx2_)
        hx2 = hx2_ + hx2
        #hx = self.activation(self.en_layer2_4(hx) + hx)
        hx3 = self.en_layer3_1(hx2)
        hx3_1, hx3_2 = hx3.split([(hx3.size()[1] // 2), (hx3.size()[1] // 2)], dim=1)
        hx3_1 = self.fusion2(hx3_1, feat_mem)
        hx3_2 = self.conv2(hx3_2)
        hx3 = torch.cat((hx3_1, hx3_2), dim=1)
        hx3_ = self.activation(self.en_layer3_2(hx3) + hx3)
        hx3_ = self.activation(self.en_layer3_3(hx3_) + hx3_)
        hx3 = hx3_ + hx3
        return hx1, hx2, hx3
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda()#.to(dtype=torch.float16)
model = Embeddings(dim=64).cuda()#.to(dtype=torch.float16)
out = model(inp)
print(out[0].shape) # 1, 64, 256, 256
print(out[1].shape) # 1, 128, 128, 128
print(out[2].shape) # 1, 256, 64, 64
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""
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

        self.fusion1 = Decoder_MDCBlock1(dim//2, 3, mode='iter2')
        self.fusion2 = Decoder_MDCBlock1(dim, 2, mode='iter2')
        
        self.conv1 = RDB(32, 4, 64)
        self.conv2 = RDB(64, 4, 128)

    def forward(self, hx1, hx2, hx3):
        in_ft = hx3*2
        
        
        hx3 = self.de_trans_level3(in_ft) + in_ft - hx3
        hx3_1, hx3_2 = hx3.split([(hx3.size()[1] // 2), (hx3.size()[1] // 2)], dim=1)
        feat_mem = [hx3_1]
        
        hx3 = self.up3_2(hx3)
        hx3 = F.upsample(hx3, hx2.size()[2:], mode='bilinear')
        hx2 = torch.add(hx3, hx2)
        hx2 = self.de_trans_level2(hx2) + hx2 - hx3 
        hx2_1, hx2_2 = hx2.split([(hx2.size()[1] // 2), (hx2.size()[1] // 2)], dim=1)
        #print(hx2_1.shape, feat_mem[-1].shape)
        hx2_1 = self.fusion2(hx2_1, feat_mem)
        hx2_2 = self.conv2(hx2_2)
        feat_mem.append(hx2_1)
        hx2 = torch.cat((hx2_1, hx2_2), dim=1)
        
        hx2 = self.up2_1(hx2)
        hx2 = F.upsample(hx2, hx1.size()[2:], mode='bilinear')
        hx1 = torch.add(hx2, hx1)
        hx1 = self.de_trans_level1(hx1) + hx1 - hx2 
        hx1_1, hx1_2 = hx1.split([(hx1.size()[1] // 2), (hx1.size()[1] // 2)], dim=1)
        hx1_1 = self.fusion1(hx1_1, feat_mem)
        hx1_2 = self.conv1(hx1_2)
        hx1 = torch.cat((hx1_1, hx1_2), dim=1)
        
        hx1 = self.refinement(hx1)
        hx1 = self.output(hx1)
        return hx1

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
class NADeblur_V11(nn.Module):
    def __init__(self,
                 dim = 64, 
                 num_blocks = [4,6,8],
                 num_refinement_blocks = 4,
                 num_heads = [2,4,8], 
                 kernel = 7, 
                 dilation = 3, 
                 ffn_expansion_factor = 1, 
                 bias = False):
        super(NADeblur_V11, self).__init__()

        #self.dwt = DWT_2D(wave='haar')
        #self.idwt = IDWT_2D(wave='haar')
        
        self.encoder = Embeddings(dim)
        #dim = 320
    
        self.decoder = Embeddings_output(dim, num_blocks, num_refinement_blocks, 
                                         num_heads, bias)


    def forward(self, x):
        
        hx, res1, res2 = self.encoder(x)

        hx = self.decoder(hx, res1, res2)

        return hx + x
"""
import time
start_time = time.time()
inp = torch.randn(1, 3, 256, 256).cuda()#.to(dtype=torch.float16)
model = NADeblur_V11().cuda()#.to(dtype=torch.float16)
out = model(inp)
print(out.shape)
print("--- %s seconds ---" % (time.time() - start_time))
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("--- {num} parameters ---".format(num = pytorch_total_params))
pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("--- {num} trainable parameters ---".format(num = pytorch_trainable_params))
"""
