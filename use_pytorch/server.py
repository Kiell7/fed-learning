import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from Models import Mnist_2NN, Mnist_CNN, LSTM_net, LSTM_QA, BERT_QA, BERT_Classifier
from clients import ClientsGroup
from utils.visualization import plot_training_progress
from pyrootutils.pyrootutils import setup_root
from torchvision import models
from transformers import ViTConfig, ViTForImageClassification
# 在文件开头导入 matplotlib
import matplotlib
matplotlib.use('Agg')  # 用于服务器环境，不需要图形界面
import matplotlib.pyplot as plt
from datetime import datetime  # 添加时间戳支持

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
parser.add_argument('-rdo', '--rdo_coef', type=float, default=500, help='rate distortion constrained optim')
parser.add_argument('-qs', '--quantizer_step', type=int, default=8, help='quantizer step')
parser.add_argument('-pt', '--pretrained', type=int, default=0, help='use pretrained model or not')
parser.add_argument('-vs', '--vocab_size', type=int, default=10000, help='vocabulary size for LSTM')
parser.add_argument('-opt', '--optimizer', type=str, default='sgd', help='optimizer type: sgd, adam, adamw')

def test_mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def create_exp_folder(args):
    """
    创建实验文件夹，使用时间戳和关键参数命名
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 构建实验名称
    exp_name = f"{args['model_name']}_{args['dataset_name']}_nc{args['num_of_clients']}_comm{args['num_comm']}"
    
    if args['backslash']:
        exp_name += f"_bs_rdo{args['rdo_coef']}"
    
    exp_name += f"_{timestamp}"
    
    # 创建实验文件夹
    exp_path = os.path.join(args['save_path'], exp_name)
    test_mkdir(exp_path)
    
    return exp_path, exp_name

if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    test_mkdir(args['save_path'])
    
    # 创建本次实验的专属文件夹
    exp_path, exp_name = create_exp_folder(args)
    print(f"实验文件夹: {exp_path}")
    
    # 保存实验配置
    import json
    config_path = os.path.join(exp_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args, f, indent=4)
    print(f"配置已保存到: {config_path}")

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # **关键修改：先加载数据集以获取实际词汇表大小**
    use_bert = (args['model_name'] == 'bert')
    myClients = ClientsGroup(args['dataset_name'], args['IID'], args['num_of_clients'], dev, use_bert=use_bert)
    testDataLoader = myClients.test_data_loader
    
    # 获取实际的词汇表大小
    if args['dataset_name'] in ['imdb', 'squad'] and not use_bert:
        actual_vocab_size = myClients.nlp_transform.vocab_size
        print(f"实际词汇表大小: {actual_vocab_size}")
    else:
        actual_vocab_size = args['vocab_size']

    # 使用实际词汇表大小初始化模型
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
                num_labels=10,
                ignore_mismatched_sizes=True
            )
        else:
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
            net = ViTForImageClassification(config)
    elif args['model_name'] == 'lstm':
        if args['dataset_name'] == 'squad':
            net = LSTM_QA(vocab_size=actual_vocab_size)
        else:
            net = LSTM_net(vocab_size=actual_vocab_size)
        net = net.to(dev)
    elif args['model_name'] == 'bert':
        # 使用 BERT 模型
        if args['dataset_name'] == 'squad':
            # SQuAD 问答任务
            net = BERT_QA(model_path='./bert_cache/bert-base-uncased-local', freeze_bert=False)
        else:
            # IMDB 或其他分类任务
            net = BERT_Classifier(num_classes=2, model_path='./bert_cache/bert-base-uncased-local', freeze_bert=False)
        net = net.to(dev)
    else:
        raise ValueError(f"Unknown model name: {args['model_name']}")

    if args["backslash"]:
        # BackSlash开始
        model_status_begin, params = get_model_status(net, return_params=True, quantizer_step=args["quantizer_step"])
        print("Before BackSlash: ", model_status_begin)

        gt = torch.load(f"{root}/tables/gamma_table.pt", weights_only=True)
        rgt = torch.load(f"{root}/tables/r_gamma_table.pt", weights_only=True)
        for _ in range(args["backslash_step"]):
            l1(net, args["rdo_coef"])

        model_status_end, params = get_model_status(net, return_params=True, quantizer_step=args["quantizer_step"])
        print("After BackSlash: ", model_status_end)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    # 定义损失函数
    if args['dataset_name'] == 'squad':
        # SQuAD 使用交叉熵损失，分别计算起始和结束位置
        def squad_loss_func(start_logits, end_logits, start_pos, end_pos):
            start_loss = F.cross_entropy(start_logits, start_pos)
            end_loss = F.cross_entropy(end_logits, end_pos)
            return (start_loss + end_loss) / 2
        loss_func = squad_loss_func
    else:
        loss_func = F.cross_entropy

    # 根据参数选择优化器
    optimizer_type = args['optimizer'].lower()
    if optimizer_type == 'sgd':
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9)
        print(f"使用优化器: SGD (lr={args['learning_rate']}, momentum=0.9)")
    elif optimizer_type == 'adam':
        opti = optim.Adam(net.parameters(), lr=args['learning_rate'])
        print(f"使用优化器: Adam (lr={args['learning_rate']})")
    elif optimizer_type == 'adamw':
        opti = optim.AdamW(net.parameters(), lr=args['learning_rate'])
        print(f"使用优化器: AdamW (lr={args['learning_rate']})")
    else:
        print(f"⚠️  未知的优化器类型: {optimizer_type}, 使用默认的SGD")
        opti = optim.SGD(net.parameters(), lr=args['learning_rate'], momentum=0.9)

    # myClients 已经在前面初始化了，这里移除重复的初始化
    # myClients = ClientsGroup(args['dataset_name'], args['IID'], args['num_of_clients'], dev)
    # testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    losses = []
    accuracies = []
    all_lengths = []  # 记录所有轮次的编码长度
    
    # 创建CSV日志文件，记录每轮详细参数
    import csv
    log_csv_path = os.path.join(exp_path, 'training_log.csv')
    csv_file = open(log_csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Round', 'Accuracy', 'Loss', 'Length_1', 'Length_2', 'Length_3'])
    print(f"训练日志将保存到: {log_csv_path}")

    # tqdm: 进度条
    for r in range(args['num_comm']):
        print("communicate round {}".format(r+1))

        order = np.random.permutation(args['num_of_clients'])
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]

        sum_parameters = None
        epoch_loss = []

        for client in tqdm(clients_in_comm):
            loss, local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            epoch_loss.append(loss)
            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()
            else:
                for var in sum_parameters:
                    sum_parameters[var] = sum_parameters[var] + local_parameters[var]

        loss_avg = sum(epoch_loss) / len(epoch_loss)

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / num_in_comm)

        net.load_state_dict(global_parameters, strict=True)
        params=get_params(net)
        lengths = cal_gradient_length(params, args["quantizer_step"])
        print(f"lengths:{lengths}")

        acc = myClients.accuracy_test(net)
        print('round: %d, acc: %.3f, loss: %.3f' % (r, acc, loss_avg))

        # 保存 checkpoint 到实验文件夹
        if (r + 1) % args['save_freq'] == 0:
            checkpoint_path = os.path.join(exp_path, f'checkpoint_round_{r+1}.pth')
            torch.save({
                'round': r + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opti.state_dict(),
                'accuracy': acc,
                'loss': loss_avg,
                'lengths': lengths
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        losses.append(loss_avg)
        accuracies.append(acc)
        all_lengths.append(lengths)
        
        # 写入CSV日志（每轮都记录）
        csv_writer.writerow([
            r + 1,
            f'{acc:.4f}',
            f'{loss_avg:.6f}',
            f'{lengths[0]:.6f}' if len(lengths) > 0 else 'N/A',
            f'{lengths[1]:.6f}' if len(lengths) > 1 else 'N/A',
            f'{lengths[2]:.6f}' if len(lengths) > 2 else 'N/A'
        ])
        csv_file.flush()  # 实时写入文件
    
    # 关闭CSV文件
    csv_file.close()
    print(f'训练日志已保存到: {log_csv_path}')

    # 训练结束后保存最终模型到实验文件夹
    final_model_path = os.path.join(exp_path, 'final_model.pth')
    torch.save({
        'round': args['num_comm'],
        'model_state_dict': net.state_dict(),
        'accuracy': accuracies[-1],
        'loss': losses[-1],
        'lengths': all_lengths[-1] if all_lengths else None
    }, final_model_path)
    print(f'Final model saved to {final_model_path}')

    # 保存训练曲线到实验文件夹（包括编码长度曲线）
    plot_path = os.path.join(exp_path, 'training_progress.png')
    plot_training_progress(losses, accuracies, all_lengths, save_path=plot_path)
    print(f"训练完成！可视化图表已保存到: {plot_path}")
