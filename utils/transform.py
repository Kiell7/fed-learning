#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fed-learning 
@File    ：transform.py
@Author  ：chen_mingyi
@Date    ：2025/10/31 22:19 
'''

from torchvision import transforms

mnist_transform = transforms.Compose([
    # 首先将单通道灰度图转换为三通道RGB图（ResNet需要3通道输入）
    transforms.Grayscale(num_output_channels=3),
    # 调整图像大小到ResNet的标准输入尺寸
    transforms.Resize((224, 224)),
    # 转换为Tensor并归一化到[0,1]
    transforms.ToTensor(),
    # 使用ImageNet的均值和标准差进行归一化（ResNet预训练权重使用这些值）
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mnist_transform_client = transforms.Compose([
    # 首先将Tensor转换为PIL Image（假设数据是numpy格式的灰度图）
    transforms.ToPILImage(),
    # 首先将单通道灰度图转换为三通道RGB图（ResNet需要3通道输入）
    transforms.Grayscale(num_output_channels=3),
    # 调整图像大小到ResNet的标准输入尺寸
    transforms.Resize((224, 224)),
    # 转换为Tensor并归一化到[0,1]
    transforms.ToTensor(),
    # 使用ImageNet的均值和标准差进行归一化（ResNet预训练权重使用这些值）
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cifar10_transform = transforms.Compose([
    # 调整图像大小到ResNet的标准输入尺寸
    transforms.Resize((224, 224)),
    # 转换为Tensor并归一化到[0,1]
    transforms.ToTensor(),
    # 使用ImageNet的均值和标准差进行归一化（ResNet预训练权重使用这些值）
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cifar10_transform_client = transforms.Compose([
    # 首先将Tensor转换为PIL Image（假设数据是numpy格式的灰度图）
    transforms.ToPILImage(),
    # 调整图像大小到ResNet的标准输入尺寸
    transforms.Resize((224, 224)),
    # 转换为Tensor并归一化到[0,1]
    transforms.ToTensor(),
    # 使用ImageNet的均值和标准差进行归一化（ResNet预训练权重使用这些值）
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])