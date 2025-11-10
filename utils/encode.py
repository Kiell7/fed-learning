import torch
import heapq
import math
from collections import Counter

"""目标：哥伦布编码"""
"""功能：int->str"""


# =======================
# 1. 节点类定义
# =======================


# 指数哥伦布编码类
class ExpGolombCode:
    def __init__(self, k=0):
        self.k = k

    def encode(self, nums):
        codes = [None] * len(nums)
        for i, num in enumerate(nums):
            code = num + (1 << self.k)
            codes[i] = "0" * (int(code).bit_length() - self.k - 1) + bin(code)[2:]
        return codes

    def encode_fast(self, nums):
        codes = nums + (1 << self.k)
        zeros = torch.ceil(torch.log2(codes)) - self.k - 1
        return zeros, codes

    def decode(self, codes):
        nums = torch.zeros(len(codes), dtype=torch.long)
        for i, code in enumerate(codes):
            num = int("0b" + code, base=2)
            nums[i] = num - (1 << self.k)
        return nums

    def streamEncode(self, nums):
        codes = self.encode(nums)
        return "".join(codes)

    def streamDecode(self, streamStr):
        codes = []
        start = 0
        while start < len(streamStr):
            cnt = 0
            while streamStr[start + cnt] == "0":
                cnt += 1
            end = start + 2 * cnt + self.k + 1
            codes.append(streamStr[start:end])
            start = end
        nums = self.decode(codes)
        return nums


# 哈夫曼树节点类
class HuffmanNode:
    def __init__(self, symbol, frequency):
        self.symbol = symbol  # 符号
        self.frequency = frequency  # 频率
        self.left = None  # 左子树
        self.right = None  # 右子树

    def __lt__(self, other):
        return self.frequency < other.frequency


# 建立哈夫曼树
def build_huffman_tree(symbols, frequencies):
    heap = [HuffmanNode(symbols[i], frequencies[i]) for i in range(len(symbols))]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        new_node = HuffmanNode(None, left.frequency + right.frequency)
        new_node.left = left
        new_node.right = right
        heapq.heappush(heap, new_node)

    return heap[0]


# 通过哈夫曼树计算哈夫曼编码的平均码长
def calculate_average_code_length(huffman_tree, frequency_dict):
    def _get_code_length(node, depth):
        if node.symbol is not None:
            return node.frequency * depth
        return _get_code_length(node.left, depth + 1) + _get_code_length(
            node.right, depth + 1
        )

    return _get_code_length(huffman_tree, 0) / sum(frequency_dict.values())


# 计算参数中连续0的个数
def count_consecutive_zeros(tensor):
    indices = torch.nonzero(tensor, as_tuple=True)[0]
    all_indices = torch.cat(
        [
            torch.tensor([-1], device=tensor.device),
            indices,
            torch.tensor([len(tensor)], device=tensor.device),
        ]
    )
    zero_lengths = all_indices[1:] - all_indices[:-1] - 1
    lengths = zero_lengths[zero_lengths > 0]
    return lengths


# =======================
# 2. 编码函数定义
# =======================


# Exp-Golomb码
def cal_eg_length(params, k=0):
    return (2 * torch.floor(torch.log2(params + 2**k)) - k + 1).mean()


# 0优化的Exp-Golomb码
def cal_eg_length_pro(params):
    zeros = count_consecutive_zeros(params)
    zeros_length = (
        2 * torch.floor(torch.log2(zeros + 1)) + 2
    )  # zeros的编码格式为[1, enc(n)]

    non_zeros = params[params != 0]
    non_zeros_length = (
        2 * torch.floor(torch.log2(non_zeros + 1)) + 1
    )  # non_zeros的编码[enc(n)]

    return (torch.sum(zeros_length) + torch.sum(non_zeros_length)) / params.numel()


# Golomb-Rice码
def cal_gr_length(params, k=0):
    return (torch.floor(params / 2**k) + k + 1).mean()


# Fixed-Length码
def cal_fix_length(params):
    return torch.ceil(torch.log2(params.max() + 1))


# Huffman码
def cal_huffman_length(params):
    symbols, frequencies = torch.unique(params, return_counts=True)
    total_frequency = sum(frequencies)
    normalized_frequencies = [f / total_frequency for f in frequencies]
    huffman_tree = build_huffman_tree(symbols, frequencies)
    return calculate_average_code_length(huffman_tree, dict(zip(symbols, frequencies)))


# AC码
def calculate_arithmetic_code_length(data):
    if torch.is_tensor(data):
        data = data.cpu().flatten().tolist()
    elif hasattr(data, "numpy"):
        data = data.numpy().flatten().tolist()
    if not data:
        return 0.0
    frequency = Counter(data)
    total_symbols = len(data)
    probabilities = {
        symbol: count / total_symbols for symbol, count in frequency.items()
    }

    # 计算平均码长
    average_length = 0.0
    for symbol, prob in probabilities.items():
        if prob > 0:
            average_length += -prob * math.log2(prob)

    return average_length
