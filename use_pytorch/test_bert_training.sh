#!/bin/bash
# BERT 快速测试配置 - 用于验证训练流程

cd /home/student4/njj/fed-learning-feature-dev/use_pytorch

echo "========================================="
echo "BERT 快速测试 (小规模配置)"
echo "========================================="
echo ""
echo "配置:"
echo "  模型: BERT-base-uncased"
echo "  数据集: SQuAD 1.1"
echo "  客户端数量: 5 (快速测试)"
echo "  采样比例: 1.0 (所有客户端)"
echo "  本地epoch: 1"
echo "  批大小: 8 (减小以加快速度)"
echo "  学习率: 2e-5"
echo "  通信轮次: 5 (快速测试)"
echo ""
echo "预计时间: 约10-15分钟"
echo "========================================="
echo ""

python server.py \
    -mn bert \
    -dsn squad \
    -nc 5 \
    -cf 1.0 \
    -E 1 \
    -B 8 \
    -lr 0.00002 \
    -ncomm 5 \
    -sf 5

echo ""
echo "========================================="
echo "快速测试完成！"
echo "如果成功，运行完整训练:"
echo "  python server.py -mn bert -dsn squad -nc 20 -cf 0.5 -E 2 -B 16 -lr 0.00002 -ncomm 100 -sf 10"
echo "========================================="
