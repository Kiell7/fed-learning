# BERT 模型下载指南

## 方法1: 使用 Git 克隆 (推荐，最快)

```bash
cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
mkdir -p bert_cache
cd bert_cache

# 从清华镜像源克隆
git clone https://hf-mirror.com/bert-base-uncased bert-base-uncased-local
```

## 方法2: 手动下载文件

访问清华镜像源: **https://hf-mirror.com/bert-base-uncased**

需要下载的文件 (放到 `./bert_cache/bert-base-uncased-local/` 目录):

1. **config.json** (约1KB)
   - https://hf-mirror.com/bert-base-uncased/resolve/main/config.json

2. **pytorch_model.bin** (约440MB) - **核心权重文件**
   - https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin

3. **tokenizer_config.json** (约1KB)
   - https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer_config.json

4. **vocab.txt** (约226KB)
   - https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt

5. **tokenizer.json** (约466KB)
   - https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer.json

### 下载命令 (使用 wget):

```bash
cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
mkdir -p bert_cache/bert-base-uncased-local
cd bert_cache/bert-base-uncased-local

# 下载所有必需文件
wget https://hf-mirror.com/bert-base-uncased/resolve/main/config.json
wget https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin
wget https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer_config.json
wget https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt
wget https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer.json
```

## 方法3: 使用 Python 脚本自动下载

```bash
cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
conda activate fed_learning_njj
python download_bert.py
```

## 验证下载

下载完成后，目录结构应该是:

```
bert_cache/
└── bert-base-uncased-local/
    ├── config.json
    ├── pytorch_model.bin  (约440MB)
    ├── tokenizer_config.json
    ├── vocab.txt
    └── tokenizer.json
```

## BERT 模型信息

- **模型名称**: bert-base-uncased
- **参数量**: 约110M (1.1亿)
- **隐藏层大小**: 768
- **层数**: 12
- **注意力头数**: 12
- **词汇表大小**: 30522
- **最大序列长度**: 512
- **模型大小**: 约440MB

## 使用示例

```python
from transformers import BertModel, BertTokenizer

# 从本地加载
model_path = "./bert_cache/bert-base-uncased-local"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertModel.from_pretrained(model_path)

print("✅ BERT 模型加载成功!")
```
