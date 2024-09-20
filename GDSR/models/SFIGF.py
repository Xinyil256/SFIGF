import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from torchvision import transforms
import matplotlib.pyplot as plt

import numpy as np
import matplotlib
import os
import glob

# import sys
# sys.path.append('./')
# sys.path.append('../')

from models.arch_util import LayerNorm2d
# from inspect import isfunction
# from einops import rearrange, repeat
# from torch import nn, einsum
from models.natten2d import NeighborhoodAttention2D
from natten import NeighborhoodAttention2D as NA2D

# from ldm.modules.diffusionmodules.util import checkpoint
# from basicsr.utils.registry import ARCH_REGISTRY

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
        # self.downsample = nn.MaxPool2d(2,2)
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
        self.downsample_ig1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats, 3, padding=1, bias=True, groups =n_feats*1),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats*2, 1, padding=0, bias=True),
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
        self.downsample_ig2 = nn.Sequential(
            nn.Conv2d(n_feats*4, n_feats*2, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(n_feats*2, n_feats*2, 3, padding=1, bias=True, groups =n_feats*2),
            nn.GELU(),
            nn.Conv2d(n_feats*2, n_feats*4, 1, padding=0, bias=True),
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
        self.downsample_ig3 = nn.Sequential(
            nn.Conv2d(n_feats*8, n_feats*4, 1, padding=0, bias=True),
            nn.GELU(),
            nn.Conv2d(n_feats*4, n_feats*4, 3, padding=1, bias=True, groups =n_feats*4),
            nn.GELU(),
            nn.Conv2d(n_feats*4, n_feats*8, 1, padding=0, bias=True),
            nn.GELU(),
            nn.MaxPool2d(2,2)
        )

        
    def forward(self, inp, guide, inp_guide):
        inp_1 = self.naf_inp(torch.cat((inp, inp_guide), dim=1)) + inp#ps ch
        guide_1 = self.naf_guide(torch.cat((guide, inp_guide), dim=1)) + guide

        inp_2 = self.downsample_i1(inp_1) # ps//2
        guide_2 = self.downsample_g1(guide_1)
        inp_guide_2 = self.downsample_ig1(torch.cat((inp_1, guide_1), dim=1))

        inp_2 = self.naf_inp2(torch.cat((inp_2, inp_guide_2), dim=1)) + inp_2
        guide_2 = self.naf_guide2(torch.cat((guide_2, inp_guide_2), dim=1)) + guide_2
        
        inp_3 = self.downsample_i2(inp_2) # ps//4
        guide_3 = self.downsample_g2(guide_2)
        inp_guide_3 = self.downsample_ig2(torch.cat((inp_2, guide_2), dim=1))

        inp_3 = self.naf_inp3(torch.cat((inp_3, inp_guide_3), dim=1)) + inp_3
        guide_3 = self.naf_guide3(torch.cat((guide_3, inp_guide_3), dim=1)) + guide_3
        
        inp_4 = self.downsample_i3(inp_3) # ps//8
        guide_4 = self.downsample_g3(guide_3)
        inp_guide_4 = self.downsample_ig3(torch.cat((inp_3, guide_3), dim=1))
        inp_4 = self.naf_inp4(torch.cat((inp_4, inp_guide_4), dim=1)) + inp_4
        guide_4 = self.naf_guide4(torch.cat((guide_4, inp_guide_4), dim=1)) + guide_4


        return inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4, inp_guide, inp_guide_2, inp_guide_3, inp_guide_4
    
    

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
        inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4, inp_guide, inp_guide_2, inp_guide_3, inp_guide_4  = self.encoder(feat_rgb, feat_mono, feat_rgb_mono)
        return inp_1, guide_1, inp_2, guide_2,inp_3, guide_3,inp_4, guide_4 , inp_guide, inp_guide_2, inp_guide_3, inp_guide_4




# @ARCH_REGISTRY.register()
class SFIGF(nn.Module):
    def __init__(self,ch=32):
        super(SFIGF, self).__init__()
        # self.dbf = DBF_Module()
        
        # self.smfe = SMFE()
        self.ch = ch
        self.sn = 1
        self.smfe = SMFE(n_feat=self.ch, n_layer=self.sn)
        self.sca = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.ch*2, out_channels=self.ch*2, kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True))
        self.sca2 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.ch*2*2, out_channels=self.ch*2*2, kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True))
        self.sca3 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.ch*4*2, out_channels=self.ch*4*2, kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True))
        self.sca4 = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(in_channels=self.ch*8*3, out_channels=self.ch*8*3, kernel_size=1, padding=0, stride=1,
                    groups=1, bias=True))
        self.ca_out = CPALayer(self.ch*2+1)
        self.guide1 = ConvGuidedFilter(radius=10, ch=self.ch)
        self.guide2 = ConvGuidedFilter(radius=10, ch=self.ch*2)
        self.guide3 = ConvGuidedFilter(radius=10, ch=self.ch*4)
        self.guide4 = ConvGuidedFilter(radius=10, ch=self.ch*8)
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
        self.guide_a = nn.Sequential(
            nn.Conv2d(self.ch*2 + 1, self.ch*2 + 1, 3, padding=1, padding_mode='reflect', groups=self.ch*2 +1), 
            nn.GELU(),
            nn.Conv2d(self.ch*2 + 1, self.ch*2 + 1, 3, padding=1, padding_mode='reflect', groups=self.ch*2 + 1), 
            nn.GELU(),
            nn.Conv2d(self.ch*2 +1, 1, 1 )
        )
        self.guide_b = nn.Sequential(
            nn.Conv2d(1*3, self.ch, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch , 1, 1 )
        )
        self.out = nn.Sequential(
            nn.Conv2d(1+self.ch*2, 1+self.ch*2, 3, padding=1, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(1+self.ch*2, 1, 1 )
        )
    
    def forward(self, data): #32, 128,128
        # print('do this')
        inp = data['lr'] #1
        guide = data['image'] #3
        if guide.shape[1] ==3:
            guide = 0.299 * guide[:,0:1,:,:] + 0.587 * guide[:,1:2,:,:]+ 0.114 * guide[:,2:3,:,:]
        i1, g1, i2, g2, i3, g3, i4, g4, inp_guide, inp_guide_2, inp_guide_3, inp_guide_4 = self.smfe(inp, guide)
        


        guide_a = self.guide_a(torch.cat((i1, g1, guide),dim=1))
        guide_b = self.guide_b(torch.cat((guide_a, inp, guide),dim=1)) #0.36

      
        guided1 = self.guide1(i1, g1)
        guided2 = self.guide2(i2, g2) 
        guided3 = self.guide3(i3, g3)
        guided4 = self.guide4(i4, g4) 
        
        guided4 = torch.cat((guided4, i4, g4), dim=1)

        guided4 = self.sca4(guided4) * guided4
        up4 = self.upsample4(guided4) 

        up4 = F.pad(up4, pad=(0, -up4.shape[3]+guided3.shape[3], 0, -up4.shape[2]+guided3.shape[2]), mode='reflect')
        guided3 = torch.cat((guided3, up4), dim=1)
        guided3 = self.sca3(guided3) * guided3
      
        up3 = self.upsample3(guided3)
        guided2 = torch.cat((guided2, up3),dim=1)
        guided2 = self.sca2(guided2) * guided2

        up2 = self.upsample2(guided2)
        guided1 = torch.cat((guided1, up2), dim=1)
        guided1 = self.sca(guided1) * guided1

       
        guide_init = guide_a*guide + guide_b

        guide_conv = guided1
        guide_out = self.ca_out(torch.cat((guide_init, guide_conv),dim=1))
        guide_out = self.out(guide_out)
        return guide_out
    
    

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
        self.attn_inp = NA2D(dim=ch, kernel_size=7, dilation=3, num_heads=4)
        self.norm2 = nn.LayerNorm(ch)
        self.norm_inp = nn.LayerNorm(ch)
        self.norm_inp2 = nn.LayerNorm(ch)
                          
       
    def forward(self, p,i): 
        # b = self.conv_b(torch.cat((i,p),dim=1)) + p
        inp = self.conv_a(torch.cat((i,p), dim=1))
        shortcut = inp
        inp = inp.permute(0,2,3,1)
        inp = self.attn_inp(self.norm_inp(inp)) + inp
        inp = self.mlp_inp(self.norm_inp2(inp))
        b = inp.permute((0,3,1,2)) + p
        i = i.permute(0,2,3,1)
        p = p.permute(0,2,3,1)
        # b, c, h, w = i.shape
        q = self.mlp(self.norm2(self.attn(self.norm1(p), self.norm1(i)))) 

        q = q.permute(0,3,1,2) + b

   

        return q
