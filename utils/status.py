#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fed-learning 
@File    ：status.py
@Author  ：chen_mingyi
@Date    ：2025/11/1 16:13 
'''

import torch

from pyrootutils.pyrootutils import setup_root
root = setup_root(".", ".root", pythonpath=True)

from utils.params import eva_shape_param,get_params
from utils.codelength import cal_gradient_length


def get_model_status(model, return_params=False,quantizer_step=8):
    # 获取shape对照表
    gt = torch.load(f"{root}/tables/gamma_table.pt", weights_only=True)
    rgt = torch.load(f"{root}/tables/r_gamma_table.pt", weights_only=True)
    model_status = {}

    # 评估模型参数
    shape, std, N = eva_shape_param(model, gt, rgt)
    model_status["shape"] = shape
    model_status["standard"] = std
    model_status["numel"] = N

    # 计算参数码长
    params = get_params(model)
    eg, huff, fixed = cal_gradient_length(params, quantizer_step)
    model_status["eg_length"] = eg
    model_status["huffman_length"] = huff
    model_status["fixed_length"] = fixed

    # 添加模型参数
    if return_params == True:
        return model_status, params
    else:
        return model_status