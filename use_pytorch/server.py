import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup
from pyrootutils.pyrootutils import setup_root
from torchvision import models
from transformers import ViTConfig, ViTForImageClassification

root = setup_root(".", ".root", pythonpath=True)

from utils.backslash import backslash,l1
from utils.params import eva_shape_param
from utils.codelength import cal_gradient_length
from utils.params import get_params
from utils.status import get_model_status


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='vit-base', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.001, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)")
parser.add_argument('-sf', '--save_freq', type=int, default=20, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')
parser.add_argument('-dsn', '--dataset_name', type=str, default='mnist', help='dataset name')
parser.add_argument('-bs', '--backslash', type=int, default=0, help='if use backslash or not')
parser.add_argument('-bss', '--backslash_step', type=int, default=500, help='dataset name')
parser.add_argument('-rdo', '--rdo_coef', type=int, default=500, help='rate distortion constrained optim')
parser.add_argument('-qs', '--quantizer_step', type=int, default=8, help='quantizer step')
parser.add_argument('-pt', '--pretrained', type=int, default=0, help='use pretrained model or not')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()
    elif args['model_name'] == 'resnet18':
        net = models.resnet18(pretrained=args['pretrained'])
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, 10)
    elif args['model_name'] == 'vit-base':
        if args['pretrained']:
            net = ViTForImageClassification.from_pretrained(
                "google/vit-base-patch16-224",
                num_labels=10,  # 根据您的任务修改类别数
                ignore_mismatched_sizes=True  # 忽略分类头尺寸不匹配
            )
        else:
            # 创建ViT配置
            config = ViTConfig(
                image_size=224,
                patch_size=16,
                num_channels=3,
                num_labels=10,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                initializer_range=0.02
            )
            # 创建随机初始化的ViT模型
            net = ViTForImageClassification(config)

    if args["backslash"]:
        # BackSlash开始
        model_status_begin, params = get_model_status(net, return_params=True,quantizer_step=args['quantizer_step'])
        print("Before BackSlash: ", model_status_begin)

        gt = torch.load(f"{root}/tables/gamma_table.pt", weights_only=True)
        rgt = torch.load(f"{root}/tables/r_gamma_table.pt", weights_only=True)
        for _ in range(args["backslash_step"]):
            # backslash(net, gt, rgt, 1e7)
            l1(net, args["rdo_coef"])

        model_status_end, params = get_model_status(net, return_params=True,quantizer_step=args['quantizer_step'])
        print("After BackSlash: ", model_status_end)
        # BackSlash结束

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy
    opti = optim.Adam(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup(args['dataset_name'], args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        for client in tqdm(clients_in_comm):
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        net.load_state_dict(global_parameters, strict=True)
        params=get_params(net)
        lengths = cal_gradient_length(params, args["quantizer_step"])
        print(f"lengths:{lengths}")

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    # 这里面如果是从transformers加载的模型，输出为一个特定的类，而非直接给出预测结果
                    if not isinstance(preds, torch.Tensor):
                        preds = preds.logits
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))

        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))

