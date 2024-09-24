import os,time,scipy.io
import scipy.misc as misc

import numpy as np
import rawpy
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim

from GIRNet_sca import GIRNet as GFNet

input_dir = '/home/amax/Documents/yxxxl/oridata/Sony/Sony/short/'
gt_dir = '/home/amax/Documents/yxxxl/oridata/Sony/Sony/long/'
result_dir = './result_Sony_red_raw_girnet_sca/'
model_dir = './saved_model_red_raw_girnet_sca/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GFNet(ch=32).to(device)
#get train and test IDs
train_fns = glob.glob(gt_dir + '0*.ARW')+glob.glob(gt_dir + '2*.ARW')
# train_fns = glob.glob('/home/notebook/data/group/lxy/Flash/data/SID/Sony/short/0*_00_0.1s.ARW') + glob.glob('/home/notebook/data/group/lxy/Flash/data/SID/Sony/short/2*_00_0.1s.ARW')
train_ids = []
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[0:5]))

test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = []
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[0:5]))



ps = 512 #patch size for training
save_freq = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 100
    train_ids = train_ids[0:5]
    test_ids = test_ids[0:5]

def pack_raw(raw):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32) 
    im = np.maximum(im - 512,0)/ (16383 - 512) #subtract the black level

    im = np.expand_dims(im,axis=2) 
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2,0:W:2,:], 
                       im[0:H:2,1:W:2,:],
                       im[1:H:2,1:W:2,:],
                       im[1:H:2,0:W:2,:]), axis=2)
    return out


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()


#Raw data takes long time to load. Keep them in memory after loaded.
gt_images=[None]*6000
input_images = {}
input_images['300'] = [None]*len(train_ids)
input_images['250'] = [None]*len(train_ids)
input_images['100'] = [None]*len(train_ids)
gt_raw_images=[None]*6000

g_loss = np.zeros((5000,1))



allfolders = glob.glob('./result_Sony_red_raw_girnet_sca/*0') 
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
# model = SkipNet().to(device)
model._initialize_weights()
opt = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.5)

for epoch in range(lastepoch,10001):
    if os.path.isdir("result_Sony_red_raw_girnet_sca/%04d"%epoch):
        continue    
    cnt=0
    # if epoch > 3500:
    #     for g in opt.param_groups:
    #         g['lr'] = 1e-5
  
    scheduler.step()
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00_0.1s.ARW'%train_id)
        in_path = in_files[np.random.random_integers(0,len(in_files)-1)]
        _, in_fn = os.path.split(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        gt_path = gt_files[0]
        _, gt_fn = os.path.split(gt_path)
        in_exposure =  float(in_fn[9:-5])
        gt_exposure =  float(gt_fn[9:-5])
        ratio = min(gt_exposure/in_exposure,300)
          
        st=time.time()
        cnt+=1

        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw),axis=0) *ratio
            # raw = raw * ratio
            # raw = np.clip(raw, 0, 65535.0)
            # raw = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # input_images[str(ratio)[0:3]][ind] = np.expand_dims(np.float32(raw/65535.0),axis=0)
            # input_images[str(ratio)[0:3]][ind]  = input_images[str(ratio)[0:3]][ind] *ratio
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im/65535.0),axis = 0)
            # gt_raw_images[ind] = np.expand_dims(pack_raw(gt_raw), axis=0)

         
        #crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0,W-ps)
        yy = np.random.randint(0,H-ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:,yy:yy+ps,xx:xx+ps,:]
        gt_patch = gt_images[ind][:,yy*2:yy*2+ps*2,xx*2:xx*2+ps*2,:]
        # gt_raw_patch = gt_raw_images[ind][:,yy:yy+ps,xx:xx+ps,:]
       

        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
            # gt_raw_patch = np.flip(gt_raw_patch, axis=1)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=0)
            gt_patch = np.flip(gt_patch, axis=0)
            # gt_raw_patch = np.flip(gt_raw_patch, axis=0)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0,2,1,3))
            gt_patch = np.transpose(gt_patch, (0,2,1,3))
            # gt_raw_patch = np.transpose(gt_raw_patch, (0,2,1,3))
        
        
        input_patch = np.minimum(input_patch,1.0)
        gt_patch = np.maximum(gt_patch, 0.0)
        # gt_raw_patch = np.maximum(gt_raw_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).permute(0,3,1,2).to(device)
        gt_img = torch.from_numpy(gt_patch).permute(0,3,1,2).to(device)
        # gt_raw_img = torch.from_numpy(gt_raw_patch).permute(0,3,1,2).to(device)
        guide =  gt_img[:,0,:,:].unsqueeze(dim=1)
        _,_,H,W = guide.shape
        guide = torch.cat((guide[:,:,0:H:2,0:W:2], 
                       guide[:,:,0:H:2,1:W:2],
                       guide[:,:,1:H:2,1:W:2],
                       guide[:,:,1:H:2,0:W:2]), dim=1)
        in_img = torch.cat((in_img, guide),dim=1)
        model.zero_grad()
        out_img = model(in_img)

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        opt.step()
        loss_cpu = loss.clone().cpu()
        g_loss[ind]=loss_cpu.data

        print("%d %d Loss=%.3f Time=%.3f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),time.time()-st))
        
        if epoch%save_freq==0:
            # if not os.path.isdir(result_dir + '%04d'%epoch):
            #     os.makedirs(result_dir + '%04d'%epoch)
            # output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            # output = np.minimum(np.maximum(output,0),1)
            
            # temp = np.concatenate((gt_patch[0,:,:,:], output[0,:,:,:]),axis=1)
            # temp = temp.transpose(1,2,0)
            # temp = temp[:,:,::-1]*255
            # temp = np.clip(temp,0,255)
            # cv2.imwrite(result_dir + '%04d/%05d_00_train_%d.png'%(epoch,train_id,ratio), temp)
            # state = {'epoch': epoch,
            # 'model': model.state_dict(),
            # 'optim': optim.state_dict()}
            # torch.save(state, model_dir+'checkpoint_sony_e%04d.pth'%epoch)
            torch.save(model.state_dict(), model_dir+'checkpoint_sony_e%04d.pth'%epoch)

