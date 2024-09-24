#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2020-02-16 19:22:41
LastEditTime: 2021-01-19 20:55:10
@Description: file content
'''
from os.path import join
from torchvision.transforms import Compose, ToTensor
from .mff_dataset import Data, Data_test, Data_eval
from torchvision import transforms
import torch, numpy  #h5py, 
import torch.utils.data as data

def transform():
    return Compose([
        ToTensor(),
    ])
    
def get_data(cfg, dir):
    return Data(cfg, dir)
    
def get_test_data(cfg, dir):

    return Data_test(cfg, dir)

def get_eval_data(cfg, dir):
    return Data_eval(cfg, dir)