import numpy as np
import cv2
import utils
import glob
import os
from PIL import Image
from numpy import random
import tqdm
# np = glob.glob(r'E:\NF-F\Results\SID\Inp/' + '10228_00_0.1s.ARW')
gt_path = '/home/notebook/data/group/lxy/Flash/data/mit5k/Norm4500/'
inp_path = '/home/notebook/data/group/lxy/Flash/data/mit5k/Low4500/'
import tqdm
def image_read(gt_path, inp_path, gt_inp_list):
    """
    load image data to CPU ram, our dataset cost about 30Gb ram for training.
    if you don't have enough ram, just move this "image_read" operation to "load_data"
    it will read images from path in patch everytime.

    input: (color raw images' path list, mono raw images' path list, RGB GT images' path list)
    output: datalist
    """
    gt_list = []
    inp_list = []
    
    for i in tqdm.tqdm(range(len(gt_inp_list))):
        # print(gt_list[i])
    # for i in tqdm.tqdm(range(1)):
        # color_raw = imageio.imread(train_c_path[i])
        # inp_list.append(color_raw)
        # mono_raw = imageio.imread(train_m_path[i])
        # gt_m_list.append(mono_raw)
        # gt_rgb = imageio.imread(train_rgb_path[i])
        # gt_list.append(gt_rgb)

        inp = os.path.join(inp_path, gt_inp_list[i] )
        # print(gt_inp_list[i])
        inp = Image.open(inp)
        inp = np.asarray(inp)
        # inp = np.asarray(inp)
        inp_list.append(inp) #h,w,c
        
        gt = os.path.join(gt_path, gt_inp_list[i])
        gt = Image.open(gt)
        gt = np.asarray(gt)
        gt_list.append(gt)

    return inp_list, gt_list


gt_list = os.listdir(gt_path)

inp_list, gt_list= image_read(gt_path, inp_path,gt_list[1000:1100])
save_dir = '/home/notebook/data/group/lxy/Flash/Abandon_Bayer-Filter_See_in_the_Dark-master_0720/Abandon_Bayer-Filter_See_in_the_Dark-master/result_mit_gf/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx in range(len(inp_list)):
# for idx in range(1):
    gt_rgb_image = gt_list[idx]
    name = os.path.basename(os.listdir(gt_path)[1000+idx])
    name  = os.path.splitext(name)[0]
    # print(gt_rgb_image)
    gt_rgb_image = gt_rgb_image / 255.
    # gt_rgb_image = np.array(gt_rgb_image, dtype=np.float32)
    # print(type(gt_rgb_image))
    # gt_rgb_image = gt_rgb_image.astype(np.float32)
    # print(type(gt_rgb_image))
    inp_raw_image = inp_list[idx]
    inp_raw_image = inp_raw_image / 255.
    # inp_raw_image = np.asarray(inp_raw_image).astype(np.float32)
    var = random.randint(100,2000)/10000
    gauss = utils.generateGaussNoise(inp_raw_image, 0, var)
    noise_im = utils.validate_im( inp_raw_image + gauss )
    # print(noise_im)
    noise_im = np.clip(noise_im, 0, 1)
    H, W = inp_raw_image.shape[0:2]

    # if img_num < 500:
    #     gt_expo = 12287
    # else:
    #     gt_expo = 1023
    # print(type(gt_rgb_image))
    # print(type(inp_raw_image))
    amp = np.mean(gt_rgb_image / (inp_raw_image+0.0001))
    inp_raw_image = (noise_im  * amp).astype(np.float32)
    
    gt_rgb_image = (gt_rgb_image).astype(np.float32)

    i = random.randint(0, (H - 256 - 2) // 2) * 2
    j = random.randint(0, (W - 256 - 2) // 2) * 2
    i = 0
    j = 0
    p = inp_raw_image[i:i + 256, j:j + 256,:]
    gt_rgb = gt_rgb_image[i:i + 256, j:j + 256, :]
        # inp_raw = inp_raw_image
        # gt_rgb = gt_rgb_image
    # gt = torch.from_numpy(np.transpose(gt_rgb, [2, 0, 1])).float()
    # inp = torch.from_numpy(np.transpose(inp_raw, [2, 0, 1])).float()
    i = gt_rgb.copy()
    i[...,1:] = gt_rgb[:,:,1:]*0
    # out = cv2.ximgproc.guidedFilter(guide=i,src=p,radius=15,eps=1e-8,dDepth=-1)

    # cv2.imwrite(save_dir+name +'_inp.png', p[:,:,::-1]*255)
    cv2.imwrite(save_dir+name +'_red.png', i[:,:,::-1]*255)


#--------------------------------------------------------------mcr-----------------------------------------
# test_c_path = np.load('./random_path_list/test/test_c_path.npy')
# test_m_path = np.load('./random_path_list/test/test_m_path.npy')
# test_rgb_path = np.load('./random_path_list/test/test_rgb_path.npy')


# import imageio
# import rawpy
# import tifffile as tiff
# from PIL import Image
# def image_read(train_c_path, train_m_path, train_rgb_path):
#     """
#     load image data to CPU ram, our dataset cost about 30Gb ram for training.
#     if you don't have enough ram, just move this "image_read" operation to "load_data"
#     it will read images from path in patch everytime.

#     input: (color raw images' path list, mono raw images' path list, RGB GT images' path list)
#     output: datalist
#     """
#     gt_list = []
#     inp_list = []
#     gt_m_list = []

#     # for i in tqdm.tqdm(range(len(train_c_path))):
#     for i in tqdm.tqdm(range(2)):
#         train_c = os.path.join('/home/notebook/data/group/lxy/Flash/data', train_c_path[i].split('/')[1],train_c_path[i].split('/')[2],train_c_path[i].split('/')[3] )
#         color_raw = imageio.imread(train_c)
#         # color_raw = tiff.imread(train_c)
#         # color_raw = cv2.applyColorMap(color_raw.astype(np.uint8), cv2.COLORMAP_INFERNO)
#         # color_raw = Image.open(train_c)
#         # color_raw = color_raw.convert('RGB')
#         # color_raw = np.array(color_raw)
#         # print(color_raw.shape)
#         inp_list.append(color_raw)

#         train_m = os.path.join('/home/notebook/data/group/lxy/Flash/data', train_m_path[i].split('/')[1],train_m_path[i].split('/')[2],train_m_path[i].split('/')[3] )
#         mono_raw = imageio.imread(train_m)
#         gt_m_list.append(mono_raw)
        
#         train_rgb = os.path.join('/home/notebook/data/group/lxy/Flash/data', train_rgb_path[i].split('/')[1],train_rgb_path[i].split('/')[2],train_rgb_path[i].split('/')[3] )
#         gt_rgb = imageio.imread(train_rgb)
#         gt_list.append(gt_rgb)

#         # print(np.max(color), mono_raw.shape, gt_rgb.shape)

#     return inp_list, gt_m_list, gt_list, train_c_path

# def raw_pack(im):
#     img_shape = im.shape
#     H = img_shape[0]
#     W = img_shape[1]
#     im = np.expand_dims(im, -1)

#     out = np.concatenate((im[0:H:2,0:W:2,:], 
#                        im[0:H:2,1:W:2,:],
#                        im[1:H:2,1:W:2,:],
#                        im[1:H:2,0:W:2,:]),-1)
#     return out

# inp_list, gt_m_list, gt_list, train_c_path = image_read(test_c_path, test_m_path, test_rgb_path)

# save_dir = '/home/notebook/data/group/lxy/Flash/Abandon_Bayer-Filter_See_in_the_Dark-master_0720/Abandon_Bayer-Filter_See_in_the_Dark-master/result_mcr_gf/'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# for idx in range(len(inp_list)):

#     gt_rgb_image = gt_list[idx]
#     gt_m_image = gt_m_list[idx]
#     inp_raw_image = inp_list[idx]

#     img_num = int(test_c_path[idx][-23:-20])
#     img_expo = int(test_c_path[idx][-8:-4],16)
#     H, W = inp_raw_image.shape

#     if img_num < 500:
#         gt_expo = 12287
#     else:
#         gt_expo = 1023
#     amp = gt_expo / img_expo

#     inp_raw_image = (inp_raw_image / 255 * amp).astype(np.float32)
#     gt_m_image = (gt_m_image / 255).astype(np.float32)
#     gt_rgb_image = (gt_rgb_image / 255).astype(np.float32)

#     inp_raw = inp_raw_image
#     gt_m = gt_m_image
#     gt_rgb = gt_rgb_image

#     # gt = torch.from_numpy((np.transpose(gt_rgb, [2, 0, 1]))).float()
#     # gt_mono = torch.from_numpy(gt_m).float().unsqueeze(0)
#     # inp = torch.from_numpy(inp_raw).float().unsqueeze(0)
# # for idx in range(1):

# #     if img_num < 500:
# #         gt_expo = 12287
# #     else:
# #         gt_expo = 1023
# #     # print(type(gt_rgb_image))
# #     # print(type(inp_raw_image))
# #     amp = np.mean(gt_rgb_image / (inp_raw_image+0.0001))
# #     inp_raw_image = (noise_im  * amp).astype(np.float32)
    
# #     gt_rgb_image = (gt_rgb_image).astype(np.float32)

# #         # gt_rgb = gt_rgb_image
# #     # gt = torch.from_numpy(np.transpose(gt_rgb, [2, 0, 1])).float()
# #     # inp = torch.from_numpy(np.transpose(inp_raw, [2, 0, 1])).float()
# #     i = gt_rgb[:,:,:1]
# #     out = np.zeros_like(gt_rgb)
#     guided = cv2.ximgproc.guidedFilter(guide=gt_m,src=inp_raw,radius=15,eps=1e-8,dDepth=-1)
#     width = int(guided.shape[1] * 2 )
#     height = int(guided.shape[0] * 2)
#     dim = (width, height)
# # resize image
#     # guided = cv2.resize(guided, dim, interpolation = cv2.INTER_AREA)
#     # guided = raw_pack(guided)
#     # out[...,0] = guided[:,:,0] *0.
#     # out[...,1] = 0.5*guided[:,:,1] + 0.5*guided[:,:,3]
#     # out[...,2] = guided[:,:,2]
#     # print(test_c_path[idx])
#     name = test_c_path[idx]
#     name = os.path.splitext(os.path.basename(name))[0]
#     print(name)
#     out = cv2.cvtColor(guided, cv2.COLOR_GRAY2RGB)
#     cv2.imwrite(save_dir+name +'_out.png', out[:,:,::-1]*255)
    # cv2.imwrite(save_dir+name +'_gt.png', gt_rgb[:,:,::-1]*255)

#---------------------------------SID------------------------------------------
