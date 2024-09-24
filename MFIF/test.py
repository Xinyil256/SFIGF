#!/usr/bin/env python
# coding=utf-8
'''
@Author: wjm
@Date: 2019-10-12 23:50:07
@LastEditTime: 2020-06-23 17:50:08
@Description: test.py
'''

from utils.config  import get_config
from solver.testsolver import Testsolver
from solver.testsolver_deng import DengTestsolver
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--option_path', type=str, default='/home/amax/Documents/yxxxl/pansharpening/option_girnet_sca.yml')
args = parser.parse_args()



if __name__ == '__main__':
    cfg = get_config(args.option_path)
    solver = DengTestsolver(cfg)
    solver.run()
    