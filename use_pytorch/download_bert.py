#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
下载 BERT 预训练模型和 tokenizer

方法1: 使用此脚本自动下载 (需要网络)
    python download_bert.py

方法2: 手动下载 (推荐，国内更快)
    访问清华镜像源:
    https://hf-mirror.com/bert-base-uncased
    
    需要下载的文件:
    1. config.json - 模型配置
    2. pytorch_model.bin - 模型权重 (约440MB)
    3. tokenizer_config.json - tokenizer配置
    4. vocab.txt - 词汇表
    5. tokenizer.json - tokenizer文件
    
    下载后放到: ./bert_cache/bert-base-uncased-local/
    
    或者使用 git clone:
    cd bert_cache
    git clone https://hf-mirror.com/bert-base-uncased bert-base-uncased-local
"""
from transformers import BertModel, BertTokenizer
import os

# 设置环境变量使用清华镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 设置缓存目录
cache_dir = "./bert_cache"
os.makedirs(cache_dir, exist_ok=True)

print("=" * 80)
print("开始下载 BERT 预训练模型 (使用清华镜像源)")
print("=" * 80)
print("\n如果下载太慢，可以手动下载:")
print("  访问: https://hf-mirror.com/bert-base-uncased")
print("  或使用: git clone https://hf-mirror.com/bert-base-uncased bert_cache/bert-base-uncased-local")
print("\n" + "=" * 80 + "\n")

# 使用 bert-base-uncased (较小，适合联邦学习)
model_name = 'bert-base-uncased'

print(f"\n1. 下载 BERT tokenizer: {model_name}")
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
print(f"✅ Tokenizer 下载完成，保存在: {cache_dir}")

print(f"\n2. 下载 BERT 模型: {model_name}")
model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
print(f"✅ 模型下载完成，保存在: {cache_dir}")

# 保存到本地以便后续使用
local_model_path = os.path.join(cache_dir, "bert-base-uncased-local")
print(f"\n3. 保存模型到本地: {local_model_path}")
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print(f"✅ 本地模型保存完成")

print("\n" + "=" * 80)
print("BERT 模型下载完成！")
print("=" * 80)
print(f"\n模型信息:")
print(f"  - 模型名称: {model_name}")
print(f"  - 缓存目录: {cache_dir}")
print(f"  - 本地路径: {local_model_path}")
print(f"  - 隐藏层大小: {model.config.hidden_size}")
print(f"  - 层数: {model.config.num_hidden_layers}")
print(f"  - 注意力头数: {model.config.num_attention_heads}")
print(f"  - 词汇表大小: {model.config.vocab_size}")
print(f"  - 最大序列长度: {model.config.max_position_embeddings}")

# 测试加载
print("\n" + "=" * 80)
print("测试从本地加载...")
print("=" * 80)
test_model = BertModel.from_pretrained(local_model_path)
test_tokenizer = BertTokenizer.from_pretrained(local_model_path)
print("✅ 成功从本地加载！")

print("\n现在你可以在代码中这样使用:")
print(f"  BertModel.from_pretrained('{local_model_path}')")
print(f"  BertTokenizer.from_pretrained('{local_model_path}')")
