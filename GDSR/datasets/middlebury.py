import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import cv2
import matplotlib.pyplot as plt

import os
import glob
import random
from PIL import Image 
import tqdm

from utils import make_coord, to_pixel_samples, visualize_2d, add_noise
import torchvision.transforms as T

class MiddleburyDataset(Dataset):
    def __init__(self, root, split='test', scale=8, augment=True, downsample='bicubic', pre_upsample=False, to_pixel=False, sample_q=None, input_size=None, noisy=False, transformation=False):
        super().__init__()
        self.root = root
        self.split = split
        self.scale = scale
        self.augment = augment
        self.downsample = downsample
        self.pre_upsample = pre_upsample
        self.to_pixel = to_pixel
        self.sample_q = sample_q
        self.input_size = input_size
        self.noisy = noisy
        self.transformation=transformation

        if self.split == 'train':
            raise AttributeError('Middlebury dataset only support test mode.')
        else:
            self.image_files = sorted(glob.glob(os.path.join(root, '*output_color*')))
            self.depth_files = sorted(glob.glob(os.path.join(root, '*output_depth*')))
            assert len(self.image_files) == len(self.depth_files)
            self.size = len(self.image_files)
        
        # self.image_files = self.image_files [1:2]
        # self.depth_files = self.depth_files [1:2]

    def __getitem__(self, idx):
        
        image_file = self.image_files[idx]
        depth_file = self.depth_files[idx]

        image = cv2.imread(image_file).astype(np.uint8) # [H, W, 3]

        depth_hr = cv2.imread(depth_file)[:,:,0].astype(np.float32) # [H, W]
        depth_min = depth_hr.min()
        depth_max = depth_hr.max()
        depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)

        # crop to make divisible
        h, w = image.shape[:2]
        h = h - h % self.scale
        w = w - w % self.scale
        image = image[:h, :w]
        depth_hr = depth_hr[:h, :w]

        
        # crop after rescale
        if self.input_size is not None:
            x0 = random.randint(0, image.shape[0] - self.input_size)
            y0 = random.randint(0, image.shape[1] - self.input_size)
            image = image[x0:x0+self.input_size, y0:y0+self.input_size]
            depth_hr = depth_hr[x0:x0+self.input_size, y0:y0+self.input_size]

        h, w = image.shape[:2]

        if self.downsample == 'bicubic':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.BICUBIC))
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.BICUBIC))
        elif self.downsample == 'nearest-right-bottom':
            depth_lr = depth_hr[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale]
            image_lr = image[(self.scale - 1)::self.scale, (self.scale - 1)::self.scale]
        elif self.downsample == 'nearest-center':
            depth_lr = np.array(Image.fromarray(depth_hr).resize((w//self.scale, h//self.scale), Image.NEAREST))
            image_lr = np.array(Image.fromarray(image).resize((w//self.scale, h//self.scale), Image.NEAREST))
        elif self.downsample == 'nearest-left-top':
            depth_lr = depth_hr[::self.scale, ::self.scale]
            image_lr = image[::self.scale, ::self.scale]
        else:
            raise NotImplementedError

        if self.noisy:
            # print(depth_lr.min(), depth_lr.max())
            depth_lr = add_noise(depth_lr, sigma=0.04)        
        # depth_min = depth_hr.min()
        # depth_max = depth_hr.max()
        # depth_hr = (depth_hr - depth_min) / (depth_max - depth_min)
        # depth_lr = (depth_lr - depth_min) / (depth_max - depth_min)
        if self.transformation:
            angle = random.uniform(0, 1)  # random rotation between -10 and 10 degrees
            translate = (random.uniform(0, 0.05), random.uniform(0, 0.05))  # random translation
            scale = random.uniform(0.9, 1.1)  # random scaling
            shear = random.uniform(0, 20)  # random shearing
            # print(f'!!!!!!!!image.shape:{image.shape}')
            affine_transform = T.Compose([
                T.ToPILImage(),
                T.RandomAffine(degrees=angle, translate=translate, scale=(scale, scale), shear=shear),
            ])
            # image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
            

            image = affine_transform(image)
            image = np.asarray(image)
            # image = (torch.from_numpy(np.array(image))/(255.)).permute(2,0,1)
            # print(f'!!!!!!!!image.shape:{image.shape}', torch.max(image), torch.min(image))
            
            # cv2.imwrite('img.png', image[::-1])

        # print(f'!!!!!!!!image.shape:{image.shape}', np.max(image), np.min(image))
        image_lr_up = (np.array(Image.fromarray(image_lr).resize((w, h), Image.BICUBIC)))

        image = image.astype(np.float32).transpose(2,0,1) / 255
        image_lr = image_lr.astype(np.float32).transpose(2,0,1) / 255 # [3, H, W]
        image_lr_up = image_lr_up.astype(np.float32).transpose(2,0,1) / 255

        image = (image - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image_lr = (image_lr - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
        image_lr_up = (image_lr_up - np.array([0.485, 0.456, 0.406]).reshape(3,1,1)) / np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        # follow DKN, use bicubic upsampling of PIL
        depth_lr_up = np.array(Image.fromarray(depth_lr).resize((w, h), Image.BICUBIC))
        # print(image_lr.shape)
        
        if self.pre_upsample:
            depth_lr = depth_lr_up

        # to tensor
        image = torch.from_numpy(image).float()
        image_lr = torch.from_numpy(image_lr).float()
        image_lr_up = torch.from_numpy(image_lr_up).float()
        depth_hr = torch.from_numpy(depth_hr).unsqueeze(0).float()
        depth_lr = torch.from_numpy(depth_lr).unsqueeze(0).float()
        depth_lr_up = torch.from_numpy(depth_lr_up).unsqueeze(0).float()


        # print(f'!!!!!!!!image.shape:{image.shape}', torch.max(image), torch.min(image))

        
            # print(f'!!!!!!!!image.shape:{image.shape}', torch.max(image), torch.min(image))
        # transform
        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                return x

            image = augment(image)
            image_lr = augment(image_lr)
            image_lr_up = augment(image_lr_up)
            depth_hr = augment(depth_hr)
            depth_lr = augment(depth_lr)
            depth_lr_up = augment(depth_lr_up)

        image = image.contiguous()
        image_lr = image_lr.contiguous()
        image_lr_up = image_lr_up.contiguous()
        depth_hr = depth_hr.contiguous()
        depth_lr = depth_lr.contiguous()
        depth_lr_up = depth_lr_up.contiguous()

        # to pixel
        if self.to_pixel:
            
            hr_coord, hr_pixel = to_pixel_samples(depth_hr)

            lr_pixel = depth_lr_up.view(-1, 1)

            if self.sample_q is not None:
                sample_lst = np.random.choice(len(hr_coord), self.sample_q, replace=False)
                hr_coord = hr_coord[sample_lst]
                hr_pixel = hr_pixel[sample_lst]
                lr_pixel = lr_pixel[sample_lst]

            cell = torch.ones_like(hr_coord)
            cell[:, 0] *= 2 / depth_hr.shape[-2]
            cell[:, 1] *= 2 / depth_hr.shape[-1]
        
            return {
                'image': image_lr_up,
                'lr_image': image_lr,
                'depth_lr_up': depth_lr_up,
                'lr': depth_lr,
                'hr': hr_pixel,
                'hr_depth': depth_hr,
                'lr_pixel': lr_pixel,
                'hr_coord': hr_coord,
                'min': depth_min,
                'max': depth_max,
                'cell': cell,
                'idx': idx,
                'lr_image_up':image_lr_up
            }   

        else:
            return {
                'image': image_lr_up,
                'lr': depth_lr,
                'depth_lr_up': depth_lr_up,
                'hr': depth_hr,
                'min': depth_min,
                'max': depth_max,
                'idx': idx,
                'lr_image_up':image_lr_up
            }


    def __len__(self):
        return self.size


if __name__ == '__main__':
    print('===== test direct bicubic upsampling =====')
    for method in ['bicubic']:
        for scale in [8]:
            print(f'[INFO] scale = {scale}, method = {method}')
            d = MiddleburyDataset(root='./data/depth_enhance/01_Middlebury_Dataset', split='test', pre_upsample=True, augment=False, scale=scale, downsample=method, noisy=False)
            rmses = []
            for i in tqdm.trange(len(d)):
                x = d[i]
                lr = ((x['lr'].numpy() * (x['max'] - x['min'])) + x['min'])
                hr = ((x['hr'].numpy() * (x['max'] - x['min'])) + x['min'])
                rmse = np.sqrt(np.mean(np.power(lr - hr, 2)))
                rmses.append(rmse)
            print('RMSE = ', np.mean(rmses))