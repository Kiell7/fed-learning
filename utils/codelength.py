import torch
from pyrootutils.pyrootutils import setup_root

root = setup_root(__file__, ".root", pythonpath=True)
from utils.encode import cal_eg_length, cal_fix_length, cal_huffman_length


def quantizer(weights, step=8):
    """Quantify"""
    rate = 2**step
    codes = torch.round(rate * weights).long()
    return codes


def resort(codes):
    "Resort"
    # 建立映射表
    eles, cnts = torch.unique(codes, return_counts=True)
    cnts, indices = torch.sort(cnts, descending=True)
    source_eles = eles[indices]
    target_eles = torch.arange(len(eles), dtype=torch.int)

    # 搜索排序
    source_eles, indices = torch.sort(source_eles)
    target_eles = target_eles[indices]

    # 查找位置
    pos = torch.searchsorted(source_eles, codes.reshape(-1))
    pos = torch.clamp(pos, 0, len(source_eles) - 1)
    mapping_eles = target_eles[pos]

    return mapping_eles


def cal_eg_length(eles, k=0):
    """Calculate the Exp-Golomb Average Code Length"""
    lengths = 2 * torch.floor(torch.log2(eles + 2**k)) - k + 1
    avg_length = lengths.mean()
    return avg_length


def cal_eg_length_rcl(eles, k=0):
    """Exp-Golomb Average Code with Running Length Encoding"""
    indices = torch.nonzero(eles, as_tuple=True)[0]
    t1 = eles[indices]
    if len(indices) == 0:
        t2 = torch.tensor([len(eles)], dtype=torch.long)
    else:
        bounds = torch.cat([torch.tensor([-1]), indices, torch.tensor([len(eles)])])
        t2 = bounds[1:] - bounds[:-1] - 1

    t = torch.cat([t1, t2])
    total_length = (2 * torch.floor(torch.log2(t + 2**k)) - k + 1).sum()
    total_length += 1  # 标志位
    avg_length = total_length / len(eles)
    return avg_length


def cal_eg_length_rcl2(eles, k=0):
    indices = torch.nonzero(eles, as_tuple=True)[0]
    t1 = eles[indices]
    if len(indices) == 0:
        t2 = torch.tensor([len(eles)], dtype=torch.long)
    else:
        bounds = torch.cat([torch.tensor([-1]), indices, torch.tensor([len(eles)])])
        t2 = bounds[1:] - bounds[:-1] - 1
    t2 = t2[t2 > 0]  # 去除0间隔

    t1l = 2 * torch.floor(torch.log2(t1 + 1)) + 1  # non_zeros的编码格式为[enc(n)]
    t2l = 2 * torch.floor(torch.log2(t2 + 1)) + 2  # zeros的编码格式为[1, enc(n)]
    avg_length = (t1l.sum() + t2l.sum()) / len(eles)
    return avg_length


def cal_gradient_length(weights, quantizer_step=16):
    """Full Process of EG Code Calculation"""
    codes = quantizer(weights, quantizer_step)
    eles = resort(codes)
    eg = cal_eg_length(eles)
    # huff = cal_huffman_length(eles)
    fixed = cal_fix_length(eles)
    lengths = [round(float(i.cpu()), 4) for i in [eg, fixed]]
    return lengths


if __name__ == "__main__":
    weights = torch.randn(1000)
    print(cal_gradient_length(weights, 8))
