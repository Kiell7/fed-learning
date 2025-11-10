#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：fed-learning 
@File    ：get_dataset.py
@Author  ：chen_mingyi
@Date    ：2025/10/31 22:16 
'''
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import transform

class GetDataset():
    def __init__(self,dataset_name,root):
        self.dataset_name = dataset_name
        self.root = root

    def get_dataset(self):
        if self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(
                root=self.root,
                train=True,
                download=True,
                transform=transform.mnist_transform_client
            )

            test_dataset = datasets.MNIST(
                root=self.root,
                train=False,
                download=True,
                transform=transform.mnist_transform
            )
        elif self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(
                root=self.root,
                train=True,
                download=False,
                transform=transform.cifar10_transform_client
            )

            test_dataset = datasets.CIFAR10(
                root=self.root,
                train=False,
                download=False,
                transform=transform.cifar10_transform
            )
        else:
            raise NotImplementedError
        return train_dataset, test_dataset

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # 输入数据（如local_data）
        self.labels = labels  # 标签（如local_label）
        self.transform = transform  # 数据转换函数

    def __len__(self):
        return len(self.data)  # 数据集长度

    def __getitem__(self, idx):
        # 获取单个样本的数据和标签
        x = self.data[idx]
        y = self.labels[idx]

        # 如果有transform，应用到数据上
        if self.transform is not None:
            x = self.transform(x)

        return x, y  # 返回处理后的数据和标签

if __name__=="__main__":
    train_dataset, test_dataset = GetDataset(dataset_name='cifar10',root='../data').get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    for x, y in train_loader:
        print(x.shape, y.shape)