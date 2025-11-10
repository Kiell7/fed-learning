#!/usr/bin/env python3
"""
测试 BERT + IMDB 配置
"""

import torch
from Models import BERT_Classifier
from get_dataset import BERTIMDBDataset

print("="*60)
print("测试 BERT + IMDB 配置")
print("="*60)

# 测试1: 创建 BERT 分类模型
print("\n1. 创建 BERT 分类模型...")
model = BERT_Classifier(
    num_classes=2,
    model_path='./bert_cache/bert-base-uncased-local',
    freeze_bert=False
)
print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")

# 测试2: 创建测试数据
print("\n2. 创建测试数据...")
texts = [
    "This movie is amazing! I loved every minute of it.",
    "Terrible film, waste of time and money."
]
labels = [1, 0]  # 1=positive, 0=negative

dataset = BERTIMDBDataset(
    texts,
    labels,
    tokenizer_path='./bert_cache/bert-base-uncased-local',
    max_length=128
)
print(f"✅ 数据集创建成功，样本数: {len(dataset)}")

# 测试3: 前向传播
print("\n3. 测试前向传播...")
sample = dataset[0]
print(f"   input_ids shape: {sample['input_ids'].shape}")
print(f"   label: {sample['label']}")

# 批处理测试
batch = torch.utils.data.DataLoader(dataset, batch_size=2)
for batch_data in batch:
    input_ids = batch_data['input_ids']
    attention_mask = batch_data['attention_mask']
    token_type_ids = batch_data['token_type_ids']
    labels = batch_data['label']
    
    print(f"   batch input_ids shape: {input_ids.shape}")
    print(f"   batch labels: {labels}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, token_type_ids)
        print(f"   batch logits shape: {logits.shape}")
        print(f"   predictions: {torch.argmax(logits, dim=1)}")
    
    break

print("\n" + "="*60)
print("✅ 所有测试通过！BERT + IMDB 配置正常")
print("="*60)
print("\n现在可以运行:")
print("python server.py -mn bert -dsn imdb -nc 20 -cf 0.1 -E 2 -B 16 -lr 0.00002 -ncomm 100 -sf 50")
