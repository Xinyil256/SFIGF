#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-23 14:57:22
LastEditTime: 2021-01-19 20:57:29
@Description: file content
'''
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
import glob

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def get_patch(image_a, image_b, image_c, patch_size, ix=-1, iy=-1):
    (ih, iw) = image_a.size
    ip = patch_size
    # (th, tw) = (scale * ih, scale * iw)

    # patch_mult = scale #if len(scale) > 1 else 1
    # tp = patch_mult * patch_size
    # ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    # (tx, ty) = (scale * ix, scale * iy)

    image_a = image_a.crop((iy,ix,iy + ip, ix + ip))
    image_b = image_b.crop((iy,ix,iy + ip, ix + ip))
    image_c = image_c.crop((iy,ix,iy + ip, ix + ip))
                
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip}

    return image_a, image_b, image_c, info_patch

def augment(image_a,image_b, fusion, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        image_a = ImageOps.flip(image_a)
        image_b = ImageOps.flip(image_b)
        fusion = ImageOps.flip(fusion)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            image_a = ImageOps.mirror(image_a)
            image_b = ImageOps.mirror(image_b)
            fusion = ImageOps.mirror(fusion)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            image_a = image_a.rotate(180)
            image_b = image_b.rotate(180)
            fusion = fusion.rotate(180)
            info_aug['trans'] = True
            
    return image_a,image_b, fusion, info_aug

class Data(data.Dataset):
    def __init__(self, cfg, data_dir, transform=None):
        super(Data, self).__init__()
        dataset = cfg['data']['name']
        if dataset == 'MFF-WHU':
            self.imagea_dir = os.path.join(data_dir, 'source_1')
            self.imageb_dir = os.path.join(data_dir, 'source_2')
            self.fusion_dir = os.path.join(data_dir, 'full_clear')
        elif dataset == 'Real-MFF':
            self.imagea_dir = os.path.join(data_dir, 'imageA')
            self.imageb_dir = os.path.join(data_dir, 'imageB')
            self.fusion_dir = os.path.join(data_dir, 'Fusion')
            self.imga_filenames = sorted([join(self.imagea_dir, x) for x in listdir(self.imagea_dir) if is_image_file(x)])
            self.imgb_filenames = sorted([join(self.imageb_dir, x) for x in listdir(self.imageb_dir) if is_image_file(x)])
            self.fusion_filenames = sorted([join(self.fusion_dir, x) for x in listdir(self.fusion_dir) if is_image_file(x)])
        elif dataset == 'Lytro':
            self.imga_filenames = sorted(glob.glob(os.path.join(data_dir,'*-A.jpg')))
            self.imgb_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))
            self.fusion_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
        self.data_num = cfg['data']['num']
        self.imga_filenames = self.imga_filenames[self.data_num:]
        self.imgb_filenames = self.imgb_filenames[self.data_num:]
        self.fusion_filenames = self.fusion_filenames[self.data_num:]
        # else:
        #     print('Wrong Dataset!!!!!')

        # self.imga_filenames = sorted([join(self.imagea_dir, x) for x in listdir(self.imagea_dir) if is_image_file(x)])
        # self.imgb_filenames = sorted([join(self.imageb_dir, x) for x in listdir(self.imageb_dir) if is_image_file(x)])
        # self.fusion_filenames = sorted([join(self.fusion_dir, x) for x in listdir(self.fusion_dir) if is_image_file(x)])

        # self.patch_size = cfg['data']['patch_size']
        # self.upscale_factor = cfg['data']['upsacle']
        # self.transform = transform
        # self.data_augmentation = cfg['data']['data_augmentation']
        # self.normalize = cfg['data']['normalize']
        # self.cfg = cfg
        # self.data_num = cfg['data']['num']
        # self.imga_filenames = self.imga_filenames[:self.data_num]
        # self.imgb_filenames = self.imgb_filenames[:self.data_num]
        # self.fusion_filenames = self.fusion_filenames[:self.data_num]

    def __getitem__(self, index):
        image_a = load_img(self.imga_filenames[index])
        image_b = load_img(self.imgb_filenames[index])
        fusion = load_img(self.fusion_filenames[index])
        _, file = os.path.split(self.imga_filenames[index])
        file = file.split('.')[0]
        # print(file)
        # _, file = os.path.split(self.imgb_filenames[index])
        # file = file.split('_')[0]
        # print(file)
        # _, file = os.path.split(self.fusion_filenames[index])
        # file = file.split('_')[0]
        # print(file)
        # print('!!!!!')
       
           
        image_a,image_b, fusion, _ = get_patch(image_a,image_b, fusion, self.patch_size)
        


        if self.data_augmentation:
            image_a,image_b, fusion, _ = augment(image_a,image_b, fusion)
        
        if self.transform:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)
            fusion = self.transform(fusion)
            # np.array(lms1, dtype=np.float32)
        image_a = torch.from_numpy(np.array(image_a,dtype=np.float32,copy=True))
        image_b = torch.from_numpy(np.array(image_b,dtype=np.float32,copy=True))
        fusion = torch.from_numpy(np.array(fusion,dtype=np.float32,copy=True))
        image_a = image_a / 255.
        image_b = image_b / 255.
        fusion = fusion / 255.
        # print('!!!!!!!!!!!!!!!!!!!', image_a.shape)
        image_a = image_a.permute(2,0,1)
        image_b = image_b.permute(2,0,1)
        fusion = fusion.permute(2,0,1)
        # if self.normalize:
        #     ms_image = ms_image * 2 - 1
        #     lms_image = lms_image * 2 - 1
        #     pan_image = pan_image * 2 - 1
        #     bms_image = bms_image * 2 - 1

        return {'image_a': image_a ,'image_b':image_b , 'fusion':fusion , 'filename':file}

    def __len__(self):
        return len(self.fusion_filenames)

class Data_test(data.Dataset):
    def __init__(self,  cfg, data_dir, transform=None):
        super(Data_test, self).__init__()
        dataset = cfg['data']['name']
        self.dataset = dataset
        if dataset == 'MFF-WHU':
            self.imagea_dir = os.path.join(data_dir, 'source_1')
            self.imageb_dir = os.path.join(data_dir, 'source_2')
            self.fusion_dir = os.path.join(data_dir, 'full_clear')
        elif dataset == 'Real-MFF':
            self.imagea_dir = os.path.join(data_dir, 'imageA')
            self.imageb_dir = os.path.join(data_dir, 'imageB')
            self.fusion_dir = os.path.join(data_dir, 'Fusion')
        
            self.imga_filenames = sorted([join(self.imagea_dir, x) for x in listdir(self.imagea_dir) if is_image_file(x)])
            self.imgb_filenames = sorted([join(self.imageb_dir, x) for x in listdir(self.imageb_dir) if is_image_file(x)])
            self.fusion_filenames = sorted([join(self.fusion_dir, x) for x in listdir(self.fusion_dir) if is_image_file(x)])
        elif dataset == 'Lytro':
            # print('!!!!!', data_dir)
            self.imga_filenames = sorted(glob.glob(os.path.join(data_dir,'*-A.jpg')))
            self.imgb_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))
            self.fusion_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
        self.data_num = cfg['data']['num']
        self.imga_filenames = self.imga_filenames[self.data_num:]
        self.imgb_filenames = self.imgb_filenames[self.data_num:]
        self.fusion_filenames = self.fusion_filenames[self.data_num:]
        
        
        


    def __getitem__(self, index):
        # print('self.imga_filenames[index]', self.imga_filenames[index])
        # print('self.imgb_filenames[index]', self.imgb_filenames[index])
        image_a = load_img(self.imga_filenames[index])
        image_b = load_img(self.imgb_filenames[index])
        fusion = load_img(self.fusion_filenames[index])
        #print(type(ms_image)) 'PIL.TiffImagePlugin.TiffImageFile'>
        #test_img=np.array(ms_image)
        #print(test_img.max())
        _, file = os.path.split(self.imga_filenames[index])
        file = file.split('.')[0]


        image_a = torch.from_numpy(np.array(image_a,dtype=np.float32,copy=True))
        image_b = torch.from_numpy(np.array(image_b,dtype=np.float32,copy=True))
        fusion = torch.from_numpy(np.array(fusion,dtype=np.float32,copy=True))
        image_a = image_a / 255.
        image_b = image_b / 255.
        fusion = fusion / 255.
        image_a = image_a.permute(2,0,1)
        image_b = image_b.permute(2,0,1)
        fusion = fusion.permute(2,0,1)

 
        return {'image_a': image_a ,'image_b':image_b , 'fusion':fusion , 'filename':file}

    def __len__(self):
        return len(self.fusion_filenames)

class Data_eval(data.Dataset):
    def __init__(self, cfg, data_dir, transform=None):
        super(Data_eval, self).__init__()
        dataset = cfg['data']['name']
        if dataset == 'MFF-WHU':
            self.imagea_dir = os.path.join(data_dir, 'source_1')
            self.imageb_dir = os.path.join(data_dir, 'source_2')
            self.fusion_dir = os.path.join(data_dir, 'full_clear')
        elif dataset == 'Real-MFF':
            self.imagea_dir = os.path.join(data_dir, 'imageA')
            self.imageb_dir = os.path.join(data_dir, 'imageB')
            self.fusion_dir = os.path.join(data_dir, 'Fusion')
        
            self.imga_filenames = sorted([join(self.imagea_dir, x) for x in listdir(self.imagea_dir) if is_image_file(x)])
            self.imgb_filenames = sorted([join(self.imageb_dir, x) for x in listdir(self.imageb_dir) if is_image_file(x)])
            self.fusion_filenames = sorted([join(self.fusion_dir, x) for x in listdir(self.fusion_dir) if is_image_file(x)])
        elif dataset == 'Lytro':
            # print('!!!!!!!!', data_dir)
            self.imga_filenames = sorted(glob.glob(os.path.join(data_dir,'*-A.jpg')))
            self.imgb_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))
            self.fusion_filenames = sorted(glob.glob(os.path.join(data_dir,'*-B.jpg')))

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg
        self.data_num = cfg['data']['num']
        self.imga_filenames = self.imga_filenames[self.data_num:]
        self.imgb_filenames = self.imgb_filenames[self.data_num:]
        self.fusion_filenames = self.fusion_filenames[self.data_num:]
        
        
        # self.imga_filenames = sorted([join(self.imagea_dir, x) for x in listdir(self.imagea_dir) if is_image_file(x)])
        # self.imgb_filenames = sorted([join(self.imageb_dir, x) for x in listdir(self.imageb_dir) if is_image_file(x)])
        # self.fusion_filenames = sorted([join(self.fusion_dir, x) for x in listdir(self.fusion_dir) if is_image_file(x)])

        # self.patch_size = cfg['data']['patch_size']
        # self.upscale_factor = cfg['data']['upsacle']
        # self.transform = transform
        # self.data_augmentation = cfg['data']['data_augmentation']
        # self.normalize = cfg['data']['normalize']
        # self.cfg = cfg
        # self.data_num = cfg['data']['num']
        # self.imga_filenames = self.imga_filenames[self.data_num:]
        # self.imgb_filenames = self.imgb_filenames[self.data_num:]
        # self.fusion_filenames = self.fusion_filenames[self.data_num:]

    def __getitem__(self, index):
        
        image_a = load_img(self.imga_filenames[index])
        image_b = load_img(self.imgb_filenames[index])
        fusion = load_img(self.fusion_filenames[index])
        #print(type(ms_image)) 'PIL.TiffImagePlugin.TiffImageFile'>
        #test_img=np.array(ms_image)
        #print(test_img.max())
        _, file = os.path.split(self.imga_filenames[index])
        file = file.split('.')[0]
        # print(file)
        # _, file = os.path.split(self.imgb_filenames[index])
        # file = file.split('_')[0]
        # print(file)
        # _, file = os.path.split(self.fusion_filenames[index])
        # file = file.split('_')[0]
        # print(file)
        # print('------------------------')

        image_a = torch.from_numpy(np.array(image_a,dtype=np.float32,copy=True))
        image_b = torch.from_numpy(np.array(image_b,dtype=np.float32,copy=True))
        fusion = torch.from_numpy(np.array(fusion,dtype=np.float32,copy=True))
       
        image_a = image_a / 255.
        image_b = image_b / 255.
        fusion = fusion / 255.
        image_a = image_a.permute(2,0,1)
        image_b = image_b.permute(2,0,1)
        fusion = fusion.permute(2,0,1)
        return {'image_a': image_a ,'image_b':image_b, 'fusion':fusion, 'filename':file}


    def __len__(self):
        return len(self.fusion_filenames)