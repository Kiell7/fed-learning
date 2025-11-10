#!/bin/bash
# BERT 模型下载脚本 (使用清华镜像源)

cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
mkdir -p bert_cache/bert-base-uncased-local
cd bert_cache/bert-base-uncased-local

echo "开始下载 BERT-base-uncased 模型文件..."
echo "========================================"

echo "1. 下载 config.json..."
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/config.json

echo "2. 下载 tokenizer_config.json..."
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer_config.json

echo "3. 下载 vocab.txt..."
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/vocab.txt

echo "4. 下载 tokenizer.json..."
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/tokenizer.json

echo "5. 下载 pytorch_model.bin (约440MB，可能需要几分钟)..."
wget -c https://hf-mirror.com/bert-base-uncased/resolve/main/pytorch_model.bin

echo ""
echo "========================================"
echo "下载完成！"
echo "========================================"
echo "文件保存在: $(pwd)"
ls -lh

echo ""
echo "验证文件："
if [ -f "pytorch_model.bin" ]; then
    echo "✅ pytorch_model.bin ($(ls -lh pytorch_model.bin | awk '{print $5}'))"
else
    echo "❌ pytorch_model.bin 缺失"
fi

if [ -f "config.json" ]; then
    echo "✅ config.json"
else
    echo "❌ config.json 缺失"
fi

if [ -f "vocab.txt" ]; then
    echo "✅ vocab.txt"
else
    echo "❌ vocab.txt 缺失"
fi

if [ -f "tokenizer.json" ]; then
    echo "✅ tokenizer.json"
else
    echo "❌ tokenizer.json 缺失"
fi
