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
import h5py

    
class DengDataset(data.Dataset):
    def __init__(self, cfg, dir):
        super(DengDataset, self).__init__()
        self.file_path = dir
        self.img_scale = cfg['data']['img_scale']

        data = h5py.File(self.file_path)  # NxCxHxW = 0x1x2x3

        print(f"loading DengDataset_train: {self.file_path} with {self.img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / self.img_scale 
        self.gt = torch.from_numpy(gt1)  # NxCxHxW: 

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / self.img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / self.img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / self.img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        # print(torch.min(image_a))
                # print(torch.max(image_a))
        # print(torch.min(self.pan), torch.max(self.pan)) 
    #####必要函数
    def __getitem__(self, index):
        return {'gt':self.gt[index, :, :, :].float(),
               'lms':self.lms[index, :, :, :].float(),
               'ms':self.ms[index, :, :, :].float(),
               'pan':self.pan[index, :, :, :].float()}


            #####必要函数
    def __len__(self):
        return self.gt.shape[0]