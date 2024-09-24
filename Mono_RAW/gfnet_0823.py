import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torchvision import transforms
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import numpy as np
import matplotlib
import os
import glob
from arch_util import LayerNorm2d
from torch import einsum
from natten2d import  NeighborhoodAttention2D
from natten import NeighborhoodAttention2D as NA2D
from timm.models.layers import DropPath
# from inspect import isfunction
# from einops import rearrange, repeat
# from torch import nn, einsum
# from natten import NeighborhoodAttention1D, NeighborhoodAttention2D
# from ldm.modules.diffusionmodules.util import checkpoint
# from basicsr.utils.registry import ARCH_REGISTRY

# from basicsr.utils.registry import ARCH_REGISTRY


# class CALayer(nn.Module):
#     """
#     Channel Attention Layer
#     parameter: in_channel
#     More detail refer to:
#     """
#     def __init__(self, channel, reduction=16):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel * reduction, 1, padding=0, bias=True),
#             nn.GELU(),
#             nn.Conv2d(channel * reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid())

#     def forward(self, x):
#         y = self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y
class Conv_LReLU_2(nn.Module):
    """
    Network component
    (Conv + LeakyReLU) × 2
    """
    def __init__(self, in_channel, out_channel):
        super(Conv_LReLU_2, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.Conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.Conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.lrelu(self.Conv1(x))
        x = self.lrelu(self.Conv2(x))
        return x

class DBF_Module(nn.Module):
    """
    This is De-Bayer_Filter Module.
    Input: color RAW images of size (1 × H × W)
    Output: monochrome images of size (1 × H × W)
    """
    def __init__(self,):
        super(DBF_Module, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.DBF_conv1 = Conv_LReLU_2(1,8)
        self.DBF_conv2 = Conv_LReLU_2(8,16)
        self.DBF_conv3 = Conv_LReLU_2(16,32)
        self.DBF_conv4 = Conv_LReLU_2(32,64)
        self.DBF_conv5 = Conv_LReLU_2(64,128)
        self.DBF_upv5 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.DBF_conv6 = Conv_LReLU_2(128,64)
        self.DBF_upv6 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.DBF_conv7 = Conv_LReLU_2(64,32)
        self.DBF_upv7 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.DBF_conv8 = Conv_LReLU_2(32,16)
        self.DBF_upv8 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.DBF_conv9 = Conv_LReLU_2(16,8)
        self.DBF_out = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        DBF_conv1 = self.DBF_conv1(x)
        DBF_pool1 = self.pool1(DBF_conv1)

        DBF_conv2 = self.DBF_conv2(DBF_pool1)
        DBF_pool2 = self.pool1(DBF_conv2)

        DBF_conv3 = self.DBF_conv3(DBF_pool2)
        DBF_pool3 = self.pool1(DBF_conv3)

        DBF_conv4 = self.DBF_conv4(DBF_pool3)
        DBF_pool4 = self.pool1(DBF_conv4)

        DBF_conv5 = self.DBF_conv5(DBF_pool4)
        DBF_up5 = self.DBF_upv5(DBF_conv5)

        DBF_concat6 = torch.cat([DBF_up5, DBF_conv4], 1)
        DBF_conv6 = self.DBF_conv6(DBF_concat6)
        DBF_up6 = self.DBF_upv6(DBF_conv6)

        DBF_concat7 = torch.cat([DBF_up6, DBF_conv3], 1)
        DBF_conv7 = self.DBF_conv7(DBF_concat7)
        DBF_up7 = self.DBF_upv7(DBF_conv7)

        DBF_concat8 = torch.cat([DBF_up7, DBF_conv2], 1)
        DBF_conv8 = self.DBF_conv8(DBF_concat8)
        DBF_up8 = self.DBF_upv8(DBF_conv8)

        DBF_concat9 = torch.cat([DBF_up8, DBF_conv1], 1)
        DBF_conv9 = self.DBF_conv9(DBF_concat9)

        DBF_out = self.lrelu(self.DBF_out(DBF_conv9))
        # print(DBF_out.shape)
        return DBF_out + x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
class CPALayer(nn.Module):
    """
    Channel Attention Layer
    parameter: in_channel
    More detail refer to:
    """
    def __init__(self, channel, reduction=2):
        super(CPALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction , 1, padding=0, bias=True), #FC
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True)) #這是標準的channel attention 也是Squeeze and excitation attention 
        
        self.conv_spa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(channel// reduction, channel // reduction, 3, padding=1, bias=True, groups = channel // reduction),
            nn.GELU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True))
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y_a = self.conv_du(y)
        y_sp = self.conv_spa(x)
        weight = y_a + y_sp
        weight = self.act(weight)
        return x * weight



class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma    
    
    
    
class Coupled_Layer(nn.Module):
    def __init__(self,
                 coupled_number,
                 n_feats,
                 kernel_size=3):
        super(Coupled_Layer, self).__init__()
        self.n_feats = n_feats
        # self.coupled_number = coupled_number
        
        self.naf_inp = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1),
            nn.GELU(),
            NAFBlock(c=n_feats )
        )
        
        self.naf_guide = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1),
            nn.GELU(),
            NAFBlock(c=n_feats )
        )
        self.naf_inp2 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*2, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*2 )
        )
        self.naf_guide2 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*2, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*2 )
        )
        self.naf_inp3 = nn.Sequential(
            nn.Conv2d(n_feats*8, n_feats*4, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*4 )
        )
        self.naf_guide3 = nn.Sequential(
            nn.Conv2d(n_feats*8, n_feats*4, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*4 )
        )
        self.naf_inp4 = nn.Sequential(
            nn.Conv2d(n_feats*16, n_feats*8, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*8 )
        )
        self.naf_guide4 =  nn.Sequential(
            nn.Conv2d(n_feats*16, n_feats*8, 1),
            nn.GELU(),
            NAFBlock(c=n_feats*8 )
        )
        self.downsample = nn.MaxPool2d(2,2)
        self.downsample_i1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*2, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
        self.downsample_g1 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*2, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
        self.downsample_i2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*4, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
        self.downsample_g2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*4, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
        self.downsample_i3 = nn.Sequential(
            nn.Conv2d(n_feats*2*2, n_feats*4*2, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
        self.downsample_g3 = nn.Sequential(
            nn.Conv2d(n_feats*2*2, n_feats*4*2, 1),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )
                                        
        # self.igconv2 = nn.Sequential(
        #     nn.Conv2d(n_feats, n_feats*2 , kernel_size=1, bias=True, padding_mode='reflect', padding=0),
        #     nn.GELU(),
        #     nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, bias=True, groups=n_feats, padding_mode='reflect', padding=1),
        #     nn.GELU())
        self.igconv2 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats*2 , kernel_size=1, bias=True, padding_mode='reflect', padding=0),
            nn.GELU(),
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=3, bias=True, groups=n_feats*2, padding_mode='reflect', padding=1),
            nn.GELU())
        self.igconv3 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*4 , kernel_size=1, bias=True, padding_mode='reflect', padding=0),
            nn.GELU(),
            nn.Conv2d(n_feats*4, n_feats*4, kernel_size=3, bias=True, groups=n_feats*4, padding_mode='reflect', padding=1),
            nn.GELU())
        self.igconv4 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*8 , kernel_size=1, bias=True, padding_mode='reflect', padding=0),
            nn.GELU(),
            nn.Conv2d(n_feats*8, n_feats*8, kernel_size=3, bias=True, groups=n_feats*8, padding_mode='reflect', padding=1),
            nn.GELU())
        
        
    def forward(self, inp, guide, inp_guide):
        inp_1 = self.naf_inp(torch.cat((inp, inp_guide), dim=1)) + inp#ps ch
        guide_1 = self.naf_guide(torch.cat((guide, inp_guide), dim=1)) + guide

        inp_2 = self.downsample_i1(inp_1) # ps//2
        guide_2 = self.downsample_g1(guide_1)
        inp_guide_2 = self.igconv2(inp_guide)
        inp_guide_2 = self.downsample(inp_guide_2)
        # print(inp_guide_2.shape)
        inp_2 = self.naf_inp2(torch.cat((inp_2, inp_guide_2), dim=1)) + inp_2
        guide_2 = self.naf_guide2(torch.cat((guide_2, inp_guide_2), dim=1)) + guide_2
        
        inp_3 = self.downsample_i2(inp_2) # ps//4
        guide_3 = self.downsample_g2(guide_2)
        inp_guide_3 = self.igconv3(inp_guide_2)
        inp_guide_3 = self.downsample(inp_guide_3)
        inp_3 = self.naf_inp3(torch.cat((inp_3, inp_guide_3), dim=1)) + inp_3
        guide_3 = self.naf_guide3(torch.cat((guide_3, inp_guide_3), dim=1)) + guide_3
        
        inp_4 = self.downsample_i3(inp_3) # ps//8
        guide_4 = self.downsample_g3(guide_3)
        inp_guide_4 = self.igconv4(inp_guide_3)
        inp_guide_4 = self.downsample(inp_guide_4)
        inp_4 = self.naf_inp4(torch.cat((inp_4, inp_guide_4), dim=1)) + inp_4
        guide_4 = self.naf_guide4(torch.cat((guide_4, inp_guide_4), dim=1)) + guide_4


        return inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4
    
    
class SMFE(nn.Module):
    def __init__(self,
                 n_feat=64,
                 n_layer=8):
        super(SMFE, self).__init__()
        self.n_layer = n_layer
        self.feat = n_feat
        self.downsample = nn.MaxPool2d(2,2)
        self.init_rgb=nn.Sequential( 
                nn.Conv2d(self.n_layer, self.feat, kernel_size=3, padding=1, padding_mode='reflect'), # in_channels, out_channels, kernel_size
                nn.GELU(),                               
                )  
        self.init_mono=nn.Sequential( 
                nn.Conv2d(self.n_layer, self.feat, kernel_size=3, padding=1, padding_mode='reflect'), # in_channels, out_channels, kernel_size
                nn.GELU(),                             
                )
        self.init_rgb_mono=nn.Sequential( 
                nn.Conv2d(self.n_layer * 2 , self.feat, kernel_size=3, padding=1, padding_mode='reflect'), # in_channels, out_channels, kernel_size
                nn.GELU(),                             
                )             
        self.encoder = Coupled_Layer(coupled_number=n_feat, n_feats=n_feat)


    def forward(self, rgb, mono):
        feat_rgb = self.init_rgb(rgb)
        feat_mono = self.init_mono(mono)
        feat_rgb_mono = self.init_rgb_mono(torch.cat((rgb,mono), dim=1))
        inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4  = self.encoder(feat_rgb, feat_mono, feat_rgb_mono)
        return inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4 

def downshuffle(var, r):
    """
    Down Shuffle function.
    Input: variable of size (1 × H × W)
    Output: down-shuffled var of size (r^2 × H/r × W/r)
    """
    b, c, h, w = var.size()
    out_channel = c * (r ** 2) #12
    out_h = h // r
    out_w = w // r
    return var.contiguous().view(b, c, out_h, r, out_w, r)\
        .permute(0, 1, 3, 5, 2, 4).contiguous().view(b,out_channel,out_h,out_w).contiguous()

def raw_pack(im):
    img_shape = im.shape
    H = img_shape[2]
    W = img_shape[3]

    out = torch.cat((im[:,:,0:H:2,0:W:2], 
                       im[:,:,0:H:2,1:W:2],
                       im[:,:,1:H:2,1:W:2],
                       im[:,:,1:H:2,0:W:2]), dim=1)
    return out

# @ARCH_REGISTRY.register()
class GFNet(nn.Module):
    def __init__(self,ch=64):
        super(GFNet, self).__init__()
        self.dbf = DBF_Module()
        
        # self.smfe = SMFE()
        self.ch = ch
        self.sn = 4
        self.smfe = SMFE(n_feat=self.ch, n_layer=self.sn)
        # self.coupled_encoder = Coupled_Encoder(n_feat=self.ch)
        # self.ca = CALayer(self.ch*2, reduction=2)
        # self.ca2 = CALayer(self.ch*2*2, reduction=2)
        # self.ca3 = CALayer(self.ch*4*2, reduction=2)
        # self.ca4 = CALayer(self.ch*8*3, reduction=2)
        self.ca = CPALayer(self.ch*2)
        self.ca2 = CPALayer(self.ch*2*2)
        self.ca3 = CPALayer(self.ch*4*2)
        self.ca4 = CPALayer(self.ch*8*3)
        self.ca_out = CPALayer(4+4)    
        self.guide1 = ConvGuidedFilter(radius=10, ch=self.ch)
        self.guide2 = ConvGuidedFilter(radius=10, ch=self.ch*2)
        self.guide3 = ConvGuidedFilter(radius=10, ch=self.ch*4)
        self.guide4 = ConvGuidedFilter(radius=10, ch=self.ch*8)
        self.up2 = nn.PixelShuffle(2)
        self.upsample4 = nn.Sequential(
            nn.Conv2d(self.ch*8*3,self.ch*4, 1, padding=0, padding_mode='reflect'),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.ch*4, self.ch*4, 3, padding=1, padding_mode='reflect', groups = self.ch*4),
            nn.GELU()
        )
        self.upsample3 = nn.Sequential(
            nn.Conv2d(self.ch*4*2,self.ch*2, 1, padding=0, padding_mode='reflect'),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.ch*2, self.ch*2, 3, padding=1, padding_mode='reflect', groups=self.ch*2),
            nn.GELU()
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(self.ch*2*2,self.ch, 1, padding=0, padding_mode='reflect'),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups = self.ch ),
            nn.GELU()
        )
        # self.res = nn.Sequential(
        #     nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect'), 
        #     nn.GELU(),
        #     nn.Conv2d(self.ch, 12, 1 )
        # )
        self.guide_res = nn.Sequential(
            nn.Conv2d(self.ch*2, self.ch*2, 3, padding=1, padding_mode='reflect', groups=self.ch*2), 
            nn.GELU(),
            nn.Conv2d(self.ch*2, 4, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            # nn.Conv2d(self.ch, 12, 1 )
        )
        self.guide_a = nn.Sequential(
            nn.Conv2d(self.ch*2 + 4, self.ch*2 + 4, 3, padding=1, padding_mode='reflect', groups=self.ch*2 + 4), 
            nn.GELU(),
            nn.Conv2d(self.ch*2 + 4, self.ch*2 + 4, 3, padding=1, padding_mode='reflect', groups=self.ch*2 + 4), 
            nn.GELU(),
            nn.Conv2d(self.ch*2 + 4, 4, 1 ),
            nn.GELU()
        )
        self.guide_b = nn.Sequential(
            nn.Conv2d(4*3, self.ch, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch , 4, 1 ),
            nn.GELU()
        )
        self.out = nn.Sequential(
            nn.Conv2d(4+4, 32, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='reflect', groups=32), 
            nn.GELU(),
            nn.Conv2d(32 , 12, 1 )
        )

        self._initialize_weights()
    
    def forward(self, x):
        denoised_raw = self.dbf(x)
        # print('denoised_mono', denoised_mono.shape)
        # inp = downshuffle(inp,2) #(1, 4, 2/h, 2/w)
        # guide = downshuffle(denoised_raw,2) #(1, 4, 2/h, 2/w)
        # # inp = self.ca(inp)
        # guide = self.ca2(guide)
        inp = downshuffle(x, 2)
        guide = downshuffle(denoised_raw, 2)
        i1, g1, i2, g2, i3, g3, i4, g4 = self.smfe(inp, guide)
        # rgb1, mono1, rgb2, mono2, rgb3, mono3, rgb4, mono4 = self.smfe(rgb1, mono1, rgb2, mono2, rgb3, mono3, rgb4, mono4)

        guide_a = self.guide_a(torch.cat((i1, g1, guide),dim=1))
        guide_b = self.guide_b(torch.cat((guide_a, inp, guide),dim=1)) #0.36

      
        guided1 = self.guide1(i1, g1) #64 #1.01
        
        guided2 = self.guide2(i2, g2) #128
        guided3 = self.guide3(i3, g3) #256
        guided4 = self.guide4(i4, g4) #512
        
        guided4 = torch.cat((guided4, i4, g4), dim=1)
        guided4 = self.ca4(guided4)
        up4 = self.upsample4(guided4) #256, 128, 128
        guided3 = torch.cat((guided3, up4), dim=1)
        guided3 = self.ca3(guided3)
        # guided3 = guided3 + up4
        up3 = self.upsample3(guided3)
        guided2 = torch.cat((guided2, up3),dim=1)
        guided2 = self.ca2(guided2)
        # guided2 = guided2 + up3
        up2 = self.upsample2(guided2)
        guided1 = torch.cat((guided1, up2), dim=1)
        guided1 = self.ca(guided1)
        # guided1 = self.guide_res(guided1)
        # guided1 = guided1 + up2
       
        guide_init = guide_a*guide + guide_b
        guide_conv = self.guide_res(guided1)
        guide_out = self.ca_out(torch.cat((guide_init, guide_conv),dim=1))
        guide_out = self.out(guide_out)

        denoised_rgb = self.up2(guide_out)

        return denoised_raw, denoised_rgb
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.normal_(0.0, 0.02)
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ConvGuidedFilter(nn.Module):
    def __init__(self, ch, radius=1):
        super(ConvGuidedFilter, self).__init__()
        self.ch = ch
        self.conv_a = nn.Sequential(nn.Conv2d(self.ch+self.ch, self.ch*1 , kernel_size=1, bias=True, padding_mode='reflect', padding=0),
                                    # norm(32),
                                    nn.GELU(),
                                    # norm(32),
                                    nn.Conv2d(self.ch, self.ch*1, kernel_size=3, bias=True, groups=self.ch, padding_mode='reflect', padding=1),
                                    nn.GELU())
        self.conv_b =  nn.Sequential(nn.Conv2d(self.ch+self.ch, self.ch*2 , kernel_size=3, bias=False, padding_mode='reflect', padding=1, groups=self.ch*2),
                                    # norm(32),
                                    nn.GELU(),
                                    # norm(32),
                                    nn.Conv2d(self.ch*2, self.ch*1, kernel_size=1, padding=0, bias=False),
                                    )
        self.norm1 = nn.LayerNorm(ch)
        self.mlp = Mlp(in_features=ch,hidden_features=ch*4,act_layer=nn.GELU,drop=0)
        self.mlp_inp= Mlp(in_features=ch,hidden_features=ch*4,act_layer=nn.GELU,drop=0)
        self.attn = NeighborhoodAttention2D(dim=ch, dilation=3, num_heads=8,kernel_size=7)
        self.attn_inp = NA2D(dim=ch, kernel_size=7, dilation=3, num_heads=8)
        self.norm2 = nn.LayerNorm(ch)
        self.norm_inp = nn.LayerNorm(ch)
        self.norm_inp2 = nn.LayerNorm(ch)
        self.drop_path = DropPath(0.3)
                            # norm(32),
                            # nn.GELU(),)
                            # norm(32),
                            # nn.Conv2d(self.ch, self.ch*1, kernel_size=3, bias=True, groups=self.ch, padding_mode='reflect', padding=1))
       
    def forward(self, p,i): 
        # b = self.conv_b(torch.cat((i,p),dim=1)) + p
        inp = self.conv_a(torch.cat((i,p), dim=1))
        shortcut = inp
        inp = inp.permute(0,2,3,1)
        inp = self.drop_path(self.attn_inp(self.norm_inp(inp)))+inp
        inp = self.drop_path(self.mlp_inp(self.norm_inp2(inp)))
        norm_p = norm_p = F.layer_norm(p, normalized_shape=p.shape[1:])
        mean_p = torch.mean(norm_p, dim=(2,3), keepdim=True)
        b = inp.permute((0,3,1,2)) + mean_p
        i = i.permute(0,2,3,1)
        p = p.permute(0,2,3,1)
        # b, c, h, w = i.shape
        q = self.mlp(self.norm2(self.drop_path(self.attn(self.norm1(p), self.norm1(i)))))
        # print(self.norm1(p))
        q = self.drop_path(q).permute(0,3,1,2) + b
        

        return q
    
    def vis_feature(self, features, root, name):
        # print('!!!!!!!!!!!!!!!', features.shape)
        # features = torch.clip(features, 0, 0.01)
        # print(features)
        dir_save = os.path.join(root, name)
        if not os.path.exists(root):
            os.makedirs(root)
        if features.shape[1] == 128:
            for b in range(features.shape[0]):
                for c in range(features.shape[1]):
                    feature = features[b,c,:,:]
                    feature = (feature-torch.min(feature)+0.0001) / (torch.max(feature)-torch.min(feature)+0.0001) 
                    feature = feature.unsqueeze(0).cpu().clone()
                    im = transforms.ToPILImage()(feature)  
                    im.save(dir_save+'_{}_{}.jpg'.format(str(b), str(c)))

    def hist(self, features, root):
        # print('!!!!!')

# # 设置matplotlib正常显示中文和负号
        # matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
        matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
# # 随机生成（10000,）服从正态分布的数据
        # data = torch.clip(features,0,0.01)
        data = np.asarray(features.cpu())
        data = data.reshape([1,-1])

        plt.hist(data[0], bins=40, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("value")
        # 显示纵轴标签
        plt.ylabel("Freq")
        # 显示图标题
        plt.title("features")
        plt.savefig(root + 'hist.jpg')
        plt.close()

# def exists(val):
#     return val is not None
# class CrossAttention(nn.Module):
#     def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = 64
#         # context_dim = default(context_dim, query_dim)

#         self.scale = dim_head ** -0.5
#         self.heads = heads

#         self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
#         self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
#         self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, query_dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context, mask=None):
#         h = self.heads

#         q = self.to_q(x)
#         # print(q.shape)
#         # context = default(context, x)
#         k = self.to_k(context)
#         v = self.to_v(context)

#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

#         sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

#         if exists(mask):
#             mask = rearrange(mask, 'b ... -> b (...)')
#             max_neg_value = -torch.finfo(sim.dtype).max
#             mask = repeat(mask, 'b j -> (b h) () j', h=h)
#             sim.masked_fill_(~mask, max_neg_value)

#         # attention, what we cannot get enough of
#         attn = sim.softmax(dim=-1)

#         out = einsum('b i j, b j d -> b i d', attn, v)
#         out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
#         return self.to_out(out)