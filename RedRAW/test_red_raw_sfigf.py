import os,time,scipy.io

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim

# from model import SeeInDark
# from net_red_nafnet import NAFNet as SkipNet
from toimage import toimage
# from net_abdnet import our_Net as SkipNet
# from skipnet_flops_sid_red import SkipNet
# from gfnet_0804 import GFNet 
# from net_svlrm_red_raw import SVLRM as GFNet
from GIRNet_sca import GIRNet_sca
from option import args 

model_nm = 'red_raw_girnet_sca_ch32'
epoch  = '9000'

# /home/amax/Documents/yxxxl/See_In_the_Dark/saved_model_red_raw_gfnet_0804
input_dir = '/home/amax/Documents/yxxxl/oridata/Sony/Sony/short/'
gt_dir = '/home/amax/Documents/yxxxl/oridata/Sony/Sony/long/'
m_path = '/home/amax/Documents/yxxxl/See_In_the_Dark/saved_model_' + model_nm + '/'
m_name = 'checkpoint_sony_e' + epoch + '.pth'
result_dir = './result_Sony'+'_' + model_nm + '/'


test_input_paths = glob.glob('/home/amax/Documents/yxxxl/oridata/Sony/Sony/short/1*_00_0.1s.ARW')
# test_input_paths = glob.glob('/home/notebook/data/group/lxy/Flash/data/SID/Sony/short/10199_00_0.1s.ARW')
# test_input_paths = glob.glob('/home/notebook/data/group/lxy/Flash/data/SID/Sony/short/')
test_gt_paths = []
# nm=0
for x in test_input_paths:
    test_gt_paths += glob.glob('/home/amax/Documents/yxxxl/oridata/Sony/Sony/long/*' + x[-17:-12] + '*.ARW')
# print(nm)
device = torch.device('cuda')
#get test IDs
# test_fns = glob.glob(gt_dir + '*.ARW')

test_ids = []
for i in range(len(test_gt_paths)):
# for i in range(39,40):
    _, test_fn = os.path.split(test_gt_paths[i])
    test_ids.append(int(test_fn[0:5]))


def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = np.maximum(raw - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out



model = GIRNet_sca(ch=32)
model_dict = torch.load( m_path + m_name ,map_location=device)
model.load_state_dict(model_dict)
model = model.to(device)
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

with torch.no_grad():
    for test_id in test_ids:
        print('test_id', test_id)
        #test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00_0.1s.ARW'%test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            _, in_fn = os.path.split(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%test_id) 
            gt_path = gt_files[0]
            _, gt_fn = os.path.split(gt_path)
            in_exposure =  float(in_fn[9:-5])
            gt_exposure =  float(gt_fn[9:-5])
            ratio = min(gt_exposure/in_exposure,300)

            raw = rawpy.imread(in_path)
            # im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # input_full = np.expand_dims(np.float32(im/65535.0),axis=0)
            # input_full = input_full * ratio
            im = raw.raw_image_visible.astype(np.float32) 
            input_full = np.expand_dims(pack_raw(im),axis=0) *ratio

            # im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)	

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im/65535.0),axis = 0)

            input_full = np.minimum(input_full,1.0)

            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            gt_img = torch.from_numpy(gt_full).permute(0,3,1,2).to(device)
            guide =  gt_img[:,0,:,:].unsqueeze(dim=1)
            _,_,H,W = guide.shape
            guide = torch.cat((guide[:,:,0:H:2,0:W:2], 
                        guide[:,:,0:H:2,1:W:2],
                        guide[:,:,1:H:2,1:W:2],
                        guide[:,:,1:H:2,0:W:2]), dim=1)


            in_img = torch.cat((in_img, guide),1)
            out_img = model(in_img)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()

            output = np.minimum(np.maximum(output,0),1)

            output = output[0,:,:,:]
            gt_full = gt_img[0,:,:,:]
            # scale_full = scale_full[0,:,:,:]
            # origin_full = scale_full
            # scale_full = scale_full*np.mean(gt_full)/np.mean(scale_full) # scale the low-light image to the same mean of the groundtruth
            
            # toimage(origin_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_ori.png'%(test_id,ratio))
            toimage(output*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '{}_{}.png'.format(test_id,epoch))
            # toimage(scale_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_scale.png'%(test_id,ratio))
            # toimage(gt_full*255,  high=255, low=0, cmin=0, cmax=255).save(result_dir + '%5d_00_%d_gt.png'%(test_id,ratio))


