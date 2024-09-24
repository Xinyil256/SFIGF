import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.arch_util import LayerNorm2d
# from utils.registry import ARCH_REGISTRY
import time
import sys
sys.path.append('.')
sys.path.append('..')
from model.natten2d import NeighborhoodAttention2D
from natten import NeighborhoodAttention2D as NA2D
# from basicsr.utils.registry import ARCH_REGISTRY
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from torchvision import transforms
import timm
import os
from timm.models.layers import DropPath
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
        
    def forward(self, inp, guide, inp_guide):
        inp_1 = self.naf_inp(torch.cat((inp, inp_guide), dim=1)) + inp#ps ch
        guide_1 = self.naf_guide(torch.cat((guide, inp_guide), dim=1)) + guide


        return inp_1, guide_1


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
        inp_1, guide_1 = self.encoder(feat_rgb, feat_mono, feat_rgb_mono)
        return inp_1, guide_1
class Net(nn.Module):
    def __init__(self, num_channels, base_filter, args=None):
        super(Net, self).__init__()
        # self.dbf = DBF_Module()
        self.sn = num_channels
        # self.criterion = criterion
        self.ch = base_filter
        # self.coupled_encoder = Coupled_Encoder(n_layer =self.sn, n_feat=self.ch)
        self.smfe = SMFE(n_feat=self.ch, n_layer=self.sn)
        self.ca_out = CPALayer(self.sn+self.sn+self.sn+self.sn)    
        self.guide1 = ConvGuidedFilter(radius=10, ch=self.ch)
        self.guide2 = ConvGuidedFilter(radius=10, ch=self.ch)
        self.guide_res = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.sn, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
        )
        self.guide_res2 = nn.Sequential(
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.sn, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
        )
        self.guide_a = nn.Sequential(
            nn.Conv2d(self.ch*2 + self.sn, self.ch*2+ self.sn, 3, padding=1, padding_mode='reflect', groups=self.ch*2 + self.sn), 
            nn.GELU(),
            nn.Conv2d(self.ch*2+ self.sn, self.ch*2+ self.sn, 3, padding=1, padding_mode='reflect', groups=self.ch*2 + self.sn), 
            nn.GELU(),
            nn.Conv2d(self.ch*2+ self.sn, self.sn, 1 )
        )
        self.guide_b = nn.Sequential(
            nn.Conv2d(self.sn*3, self.ch, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch , self.sn, 1 )
        )
        self.guide_a2 = nn.Sequential(
            nn.Conv2d(self.ch*2+ self.sn, self.ch*2+ self.sn, 3, padding=1, padding_mode='reflect', groups=self.ch*2+ self.sn), 
            nn.GELU(),
            nn.Conv2d(self.ch*2+ self.sn, self.ch*2+ self.sn, 3, padding=1, padding_mode='reflect', groups=self.ch*2+ self.sn), 
            nn.GELU(),
            nn.Conv2d(self.ch*2+ self.sn, self.sn, 1 )
        )
        self.guide_b2 = nn.Sequential(
            nn.Conv2d(self.sn*3, self.ch, 1, padding=0, padding_mode='reflect'), 
            nn.GELU(),
            nn.Conv2d(self.ch, self.ch, 3, padding=1, padding_mode='reflect', groups=self.ch), 
            nn.GELU(),
            nn.Conv2d(self.ch , self.sn, 1 )
        )
        self.out = nn.Sequential(
            nn.Conv2d(self.sn+self.sn+self.sn+self.sn, self.sn+self.sn+self.sn+self.sn, 3, padding=1, padding_mode='reflect',groups= self.sn+self.sn+self.sn+self.sn), 
            nn.GELU(),
            nn.Conv2d(self.sn+self.sn+self.sn+self.sn , self.sn, 1 )
        )
        self._initialize_weights()
    def forward(self, image_a, image_b): #32, 128,128
        # print('!!!!!!!!!!', self.ch)
        inp = image_a
        guide = image_b
        rgb1, mono1 = self.smfe(inp, guide)
        guide_a = self.guide_a(torch.cat((rgb1, mono1, guide),dim=1))
        guide_b = self.guide_b(torch.cat((guide_a, inp, guide),dim=1)) #0.36
        guided1 = self.guide1(mono1, rgb1) #64 #1.01
        guide_init = guide_a*guide + guide_b
        guide_conv = self.guide_res(guided1)
        
        guide_a2 = self.guide_a2(torch.cat((rgb1, mono1, inp),dim=1))
        guide_b2 = self.guide_b2(torch.cat((guide_a2, inp, guide),dim=1)) #0.36
        guided2 = self.guide2(rgb1, mono1) #64 #1.01
        guide_init2 = guide_a2*inp + guide_b2
        guide_conv2 = self.guide_res2(guided2)
        
        guide_out = self.ca_out(torch.cat((guide_init, guide_conv, guide_init2, guide_conv2),dim=1))
        sr_ms = self.out(guide_out)

        # denoised_rgb = self.up2(guide_out)

        return sr_ms
    
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
        self.attn_inp = NA2D(dim=ch, kernel_size=7, dilation=3, num_heads=4)
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
        b = inp.permute((0,3,1,2)) + p
        i = i.permute(0,2,3,1)
        p = p.permute(0,2,3,1)
        # b, c, h, w = i.shape
        q = self.mlp(self.norm2(self.drop_path(self.attn(self.norm1(p), self.norm1(i)))))
        # print(self.norm1(p))
        q = self.drop_path(q).permute(0,3,1,2) + b
        

        return q
    
    def vis_feature(self, features, root, name):
        # print('!!!!!!!!!!!!!!!!!!!!!!!!!')
        dir_save = os.path.join(root, name)
        if not os.path.exists(root):
            os.makedirs(root)
        # print('features.shape[1]', features.shape[1])
        if features.shape[1] == 64:
            for b in range(features.shape[0]):
                for c in range(features.shape[1]):
                    feature = features[b,c,:,:]
                    feature = (feature-torch.min(feature)+0.0001) / (torch.max(feature)-torch.min(feature)+0.0001) 
                    feature = feature.unsqueeze(0).cpu().clone()
                    im = transforms.ToPILImage()(feature)  
                    im.save(dir_save+'_{}_{}.jpg'.format(str(b), str(c)))
    
    
    
if __name__ == '__main__':

    import torch
    import torch.nn as nn
    from thop import profile
    from thop import clever_format
    import time
    import yaml
    from utils.config import get_config
# from solver.solverkw import Solver
    from solver.solver import Solver
    from solver.solver_deng import DengSolver
    import argparse
    
    parser = argparse.ArgumentParser(description='N_SR')
    parser.add_argument('--option_path', type=str, default='/home/amax/Documents/yxxxl/pansharpening/option_mff_sfigf.yml')
    opt = parser.parse_args()
    cfg = get_config(opt.option_path)
    # 使用之前定义的 SimpleCNN 模型
    model = Net(3,32,args=cfg).to('cuda')
    depth = torch.randn(1, 3, 128, 128).to('cuda')  # 假设输入是一个 3x32x32 的图像
    rgb = torch.randn(1, 3, 128, 128).to('cuda')

    # input = {'lr':depth, 'image':rgb} #3
    flops, params = profile(model, inputs=(depth,rgb,))


    print(f"Parameters: {params/1e6}M")

    print(f"FLOPs: {flops/1024**3}G")
    
    model.eval()

# 禁用梯度计算
    with torch.no_grad():
        # 执行一次前向传播来进行热身
        model(depth, rgb)

        # 测量前向传播时间
        start_time = time.time()
        
        # 多次前向传播，取平均时间
        num_iterations = 100
        for _ in range(num_iterations):
            output = model(depth, rgb)
        
        end_time = time.time()
        
        # 计算平均时间
        avg_time = (end_time - start_time) / num_iterations

        print(f"平均测试时间: {avg_time:.6f} 秒")
