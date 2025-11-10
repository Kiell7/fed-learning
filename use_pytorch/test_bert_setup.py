#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
测试 BERT 模型和数据加载
"""
import torch
from Models import BERT_QA
from get_dataset import GetDataset, BERTSQuADDataset
from torch.utils.data import DataLoader

print("=" * 80)
print("测试 BERT 模型和 SQuAD 数据加载")
print("=" * 80)

# 1. 测试 BERT 模型加载
print("\n1. 加载 BERT 模型...")
try:
    model = BERT_QA(model_path='./bert_cache/bert-base-uncased-local', freeze_bert=False)
    print("✅ BERT 模型加载成功!")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
except Exception as e:
    print(f"❌ BERT 模型加载失败: {e}")
    exit(1)

# 2. 测试数据集加载
print("\n2. 加载 SQuAD 数据集...")
try:
    dataset = GetDataset('squad', '../data', use_bert=True)
    train_dataset, test_dataset = dataset.get_dataset()
    print(f"✅ 数据集加载成功!")
    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    exit(1)

# 3. 测试数据样本
print("\n3. 检查数据样本...")
try:
    sample = train_dataset[0]
    print(f"  样本键: {sample.keys()}")
    print(f"  input_ids 形状: {sample['input_ids'].shape}")
    print(f"  attention_mask 形状: {sample['attention_mask'].shape}")
    print(f"  token_type_ids 形状: {sample['token_type_ids'].shape}")
    print(f"  start_pos: {sample['start_pos'].item()}")
    print(f"  end_pos: {sample['end_pos'].item()}")
    print("✅ 数据样本格式正确!")
except Exception as e:
    print(f"❌ 数据样本检查失败: {e}")
    exit(1)

# 4. 测试模型前向传播
print("\n4. 测试模型前向传播...")
try:
    # 创建一个小batch
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        start_pos = torch.stack([item['start_pos'] for item in batch])
        end_pos = torch.stack([item['end_pos'] for item in batch])
        return input_ids, attention_mask, token_type_ids, start_pos, end_pos
    
    dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    input_ids, attention_mask, token_type_ids, start_pos, end_pos = batch
    
    print(f"  Batch input_ids 形状: {input_ids.shape}")
    
    model.eval()
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)
    
    print(f"  start_logits 形状: {start_logits.shape}")
    print(f"  end_logits 形状: {end_logits.shape}")
    
    pred_start = torch.argmax(start_logits, dim=1)
    pred_end = torch.argmax(end_logits, dim=1)
    print(f"  预测 start: {pred_start}")
    print(f"  预测 end: {pred_end}")
    print(f"  真实 start: {start_pos}")
    print(f"  真实 end: {end_pos}")
    
    print("✅ 模型前向传播成功!")
except Exception as e:
    print(f"❌ 模型前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ 所有测试通过！可以开始训练了")
print("=" * 80)
print("\n运行训练:")
print("  bash run_bert_squad.sh")
print("或者:")
print("  python server.py -mn bert -dsn squad -nc 20 -cf 0.5 -E 2 -B 16 -lr 0.00002 -ncomm 100 -sf 10")
