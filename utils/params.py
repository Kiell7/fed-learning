import torch
from torch.nn import init
from torch.special import gammaln
from scipy.special import gamma as Gamma
from scipy.stats import gennorm

import numpy as np


def get_params(net):
    """从网络拿出参数"""
    params = []
    for param in net.parameters():
        params.append(param.flatten().to(torch.device("cpu")))
    params = torch.cat(params).detach()
    return params


def gg_param(param):
    "计算广义高斯参数"
    gt = torch.load("/mnt/hbnas/home/wujun/data/gamma_table.pt", weights_only=True)
    rgt = torch.load("/mnt/hbnas/home/wujun/data/r_gamma_table.pt", weights_only=True)
    param = param.cpu()

    n = param.shape[0]
    var = torch.sum(torch.pow(param, 2))
    mean = torch.sum(torch.abs(param))

    r_gamma = (n * var / mean**2).to(device=torch.device("cpu"))
    pos = torch.argmin(torch.abs(r_gamma - rgt))
    shape = gt[pos]

    scale = torch.sqrt(var) * torch.exp(
        0.5 * (gammaln(1.0 / shape) - gammaln(3.0 / shape))
    )

    return shape, scale


def gg_entropy(alpha, beta):
    """计算广义高斯熵"""
    # alpha是尺度参数，beta是形状参数
    ln_gamma = gammaln(1.0 / beta)
    entropy = (1.0 / beta) + torch.log(2.0 * alpha) + ln_gamma - torch.log(beta)
    return entropy


def eva_shape_param(model, gamma_table, r_gamma_table):
    """根据网络参数估计形状参数"""
    n, var, mean = 0, 0, 0
    for param in model.parameters():
        param = param.flatten().detach()
        n += param.shape[0]
        var += torch.sum(torch.pow(param, 2))
        mean += torch.sum(torch.abs(param))
    r_gamma = (n * var / mean**2).to(device=torch.device("cpu"))
    pos = torch.argmin(torch.abs(r_gamma - r_gamma_table))

    shape = gamma_table[pos]
    std = torch.sqrt(var / n)
    n = torch.tensor(n)
    return shape, std, n


def eva_loss(model, dataloader, device=torch.device("cuda:0")):
    """计算损失"""
    model.to(device)
    lossList = []
    for data in dataloader:
        with torch.no_grad():
            texts, masks, labels = (
                data["input_ids"].to(device),
                data["attention_mask"].to(device),
                data["labels"].to(device),
            )
            loss = model(texts, masks, labels)["loss"]
            lossList.append(loss)
    return float(sum(lossList).cpu() / len(lossList))


def gg_pdf(x, nu, std, mu):
    """广义高斯拟合"""
    gamma1 = torch.tensor(Gamma(1 / nu))
    gamma3 = torch.tensor(Gamma(3 / nu))
    e = torch.tensor(torch.e)

    gamma = 1 / std * torch.sqrt(gamma3 / gamma1)
    c1 = nu * gamma / (2 * gamma1)
    c2 = torch.pow(gamma, nu)
    y = c1 * torch.pow(e, -c2 * torch.pow(torch.abs(x - mu), nu))
    return y


def gaussian_pdf(x, std, mu):
    """高斯拟合"""
    e = torch.tensor(torch.e)
    pi = torch.tensor(torch.pi)
    c1 = 1 / (torch.sqrt(2 * pi) * std)
    c2 = 1 / (2 * std**2)
    y = c1 * torch.pow(e, -c2 * torch.pow(torch.abs(x - mu), 2))
    return y


def params_init(model, shape=2):
    """参数初始化"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            param_device = param.device
            param_dtype = param.dtype
            if len(param.shape) == 2:
                n_dim = param.shape[0]
                alpha = np.sqrt(2 / n_dim * Gamma(1 / shape) / Gamma(3 / shape))
                gennorm_params = gennorm.rvs(
                    shape, loc=0, scale=alpha, size=param.shape
                )
                param.data = torch.from_numpy(gennorm_params)
            else:
                if "weight" in name:
                    param.data = torch.ones(param.shape)
                elif "bias" in name:
                    param.data = torch.zeros(param.shape)

            param.data = param.data.to(param_dtype).to(param_device)
