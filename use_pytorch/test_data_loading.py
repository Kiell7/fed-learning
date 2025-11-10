import torch
from get_dataset import GetDataset

# 测试数据加载
dataset = GetDataset('imdb', '../data')
train_dataset, test_dataset = dataset.get_dataset()

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 检查第一个样本
sample_data, sample_label = train_dataset[0]
print(f"\n第一个样本:")
print(f"数据类型: {type(sample_data)}")
print(f"标签类型: {type(sample_label)}")
print(f"标签值: {sample_label}")

if isinstance(sample_data, str):
    print(f"文本长度: {len(sample_data)}")
    print(f"文本预览: {sample_data[:200]}")
elif isinstance(sample_data, torch.Tensor):
    print(f"Tensor形状: {sample_data.shape}")
    print(f"Tensor样例值: {sample_data[:20]}")