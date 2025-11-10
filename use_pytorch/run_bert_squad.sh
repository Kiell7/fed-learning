#!/bin/bash
# 使用 BERT 在 SQuAD 数据集上进行联邦学习训练

cd /home/student4/njj/fed-learning-feature-dev/use_pytorch

echo "========================================="
echo "BERT + SQuAD 联邦学习训练"
echo "========================================="
echo ""
echo "配置:"
echo "  模型: BERT-base-uncased (110M 参数)"
echo "  数据集: SQuAD 1.1"
echo "  客户端数量: 20"
echo "  采样比例: 0.5 (每轮10个客户端)"
echo "  本地epoch: 2"
echo "  批大小: 16 (BERT需要较小batch)"
echo "  学习率: 2e-5 (BERT推荐)"
echo "  通信轮次: 100"
echo ""
echo "========================================="
echo ""

# 激活环境
source activate fed_learning_njj

# 运行训练
python server.py \
    -mn bert \
    -dsn squad \
    -nc 20 \
    -cf 0.5 \
    -E 2 \
    -B 16 \
    -lr 0.00002 \
    -ncomm 100 \
    -sf 10

echo ""
echo "========================================="
echo "训练完成！"
echo "========================================="
