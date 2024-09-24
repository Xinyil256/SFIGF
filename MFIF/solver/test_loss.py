import cv2

import os

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
img1 = cv2.imread('/home/ubuntu/yxxxl2/oridata/RealMFF/imageA/709_A.png')    
img2 = cv2.imread('/home/ubuntu/yxxxl2/oridata/RealMFF/imageB/709_B.png')   
img_f = cv2.imread('/home/ubuntu/yxxxl2/GDSR/pansharpening/results/test/Real-MFF/mff_gfnet_us_wl1/test/709_pre.png' )
    
# img1 = torch.from_numpy(img1)
# img1 = img1.permute(2,0,1).unsqueeze(0)
# img2 = torch.from_numpy(img2)
# img2 = img2.permute(2,0,1).unsqueeze(0)  

# print(img1.shape)
# def unloss_in(img1, img2):
#     scale_1 = torch.ones((img1.shape))
#     # scale_2 = torch.ones((img1.shape))
#     # print(scale_1.shape)
#     for i in range(img1.shape[1]):
#         w_gauss = torch.Tensor(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).reshape(1, 1, 3, 3))
#         conv1 = nn.Conv2d(1, 1, (3, 3), padding=1)
#         conv1.weight = nn.Parameter(w_gauss)
#         lf_1 = conv1(img1[:,i:i+1,:,:].to(torch.float32)).detach()
#         hf_1 = torch.abs(img1[:,i:i+1,:,:] - lf_1)
#         lf_2 = conv1(img2[:,i:i+1,:,:].to(torch.float32)).detach()
#         hf_2 = torch.abs(img2[:,i:i+1,:,:] - lf_2)
#         scale_1[:,i:i+1,:,:] = torch.sgn(hf_1 - torch.minimum(hf_1, hf_2))
#     scale_2 = 1- scale_1
    
    
#     return scale_1, scale_2


# s1, s2 = unloss_in(img1=img1, img2=img2)
# print(s1.shape, s2.shape)
# s1 = np.asarray(s1[0].permute(1,2,0))
# cv2.imwrite('s1.jpg', s1[:,:,0]*255)
# s2 = np.asarray(s2[0].permute(1,2,0))
# cv2.imwrite('s2.jpg', s2[:,:,0]*255)

# def gradient(x, d):
#     smooth_kernel_x = torch.reshape(torch.FloatTensor([[0, 0], [-1, 1]]), [1, 1, 2, 2]).to(torch.device('cuda'))
#     smooth_kernel_y = torch.transpose(smooth_kernel_x, dim1=2, dim0=3)
#     if d=='x':
#         kernel = smooth_kernel_x
#     if d=='y':
#         kernel = smooth_kernel_y
#     gradient_orig = torch.abs(F.conv2d(input=x,weight=kernel, stride=1, padding=1))
#     grad_min = torch.min(gradient_orig)
#     grad_max = torch.max(gradient_orig)
#     grad_norm = torch.div((gradient_orig-grad_min), (grad_max-grad_min+0.0001))
#     grad_norm = grad_norm[:,:,:x.shape[2], :x.shape[3]]
#     return grad_norm



# im = torch.from_numpy(img_f).to(torch.device('cuda'))
# im = im.permute(2,0,1).float()
# for i in range(3):
#     im_x = gradient(im[None,i:i+1,:,:], 'x')
#     im_y = gradient(im[None,i:i+1,:,:], 'y')
#     im_x = np.asarray(im_x[0,0].cpu())
#     im_y = np.asarray(im_y[0,0].cpu())

#     cv2.imwrite('/home/ubuntu/yxxxl2/GDSR/pansharpening/solver/'+'709_{}_x.jpg'.format(str(i)), im_x*255)
#     cv2.imwrite('/home/ubuntu/yxxxl2/GDSR/pansharpening/solver/'+'709_{}_y.jpg'.format(str(i)), im_y*255)
#     cv2.imwrite('/home/ubuntu/yxxxl2/GDSR/pansharpening/solver/'+'709_{}_xy.jpg'.format(str(i)), (im_y+im_x)*255)
 
