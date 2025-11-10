# 使用 BERT 在 SQuAD 数据集上进行联邦学习

## ✅ 准备工作已完成

1. ✅ BERT 模型已下载 (bert-base-uncased, 110M 参数)
2. ✅ BERTSQuADDataset 已实现 (使用 BERT tokenizer)
3. ✅ BERT_QA 模型已实现 (基于预训练 BERT)
4. ✅ 所有测试通过

## 🚀 开始训练

### 方法1: 使用脚本 (推荐)

```bash
cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
bash run_bert_squad.sh
```

### 方法2: 直接命令

```bash
cd /home/student4/njj/fed-learning-feature-dev/use_pytorch
conda activate fed_learning_njj

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
```

## 📊 训练配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `-mn bert` | bert | 使用 BERT 模型 (110M 参数) |
| `-dsn squad` | squad | SQuAD 1.1 数据集 |
| `-nc 20` | 20 | 20个客户端 (每个约4380样本) |
| `-cf 0.5` | 0.5 | 每轮50%客户端参与 (10个) |
| `-E 2` | 2 | 每个客户端本地训练2个epoch |
| `-B 16` | 16 | 批大小16 (BERT推荐小batch) |
| `-lr 0.00002` | 2e-5 | 学习率 (BERT标准) |
| `-ncomm 100` | 100 | 100轮通信 |
| `-sf 10` | 10 | 每10轮保存一次 |

### 为什么这样配置？

1. **nc=20, cf=0.5**: 
   - 每轮10个客户端 × 4380样本 = 43,800样本 (50%数据覆盖)
   - 比 LSTM 的 nc=100, cf=0.1 (10%覆盖) 高5倍

2. **E=2, B=16**:
   - BERT 参数多 (110M vs LSTM 7M)
   - 小epoch + 小batch 避免过拟合

3. **lr=2e-5**:
   - BERT 预训练模型标准学习率
   - 比 LSTM (1e-3) 小50倍

## 🎯 预期效果

### LSTM 结果 (参考)
- 100轮: EM=2.37%, F1=6.05%
- Loss=0.210

### BERT 预期 (更好)
- **前20轮**: Loss 快速下降, EM < 5%
- **20-50轮**: EM 提升至 10-20%
- **50-100轮**: EM 稳定在 25-40%

**原因**: BERT 有预训练知识，理解语义更好

## 📈 监控训练

训练会实时显示:
```
communicate round 1
Client 0: [BERT] input_ids形状=torch.Size([16, 384])
start_pos范围: 45-256, end_pos范围: 47-280
...
round: 1, acc: 3.45, loss: 4.234
EM: 3.45%, F1: 8.21%
```

## 💾 结果保存

结果保存在:
```
./checkpoints/bert_squad_nc20_comm100_<timestamp>/
├── config.json              # 训练配置
├── checkpoint_round_10.pth  # 第10轮checkpoint
├── checkpoint_round_20.pth
├── ...
├── final_model.pth          # 最终模型
└── training_progress.png    # 训练曲线
```

## 🔧 调优建议

如果训练效果不理想:

1. **增加数据覆盖**:
   ```bash
   -nc 10 -cf 1.0  # 每轮所有客户端 (100%覆盖)
   ```

2. **调整学习率**:
   ```bash
   -lr 0.00003  # 稍微提高
   ```

3. **冻结BERT层** (只训练QA层):
   修改 `server.py` 第147行:
   ```python
   net = BERT_QA(model_path='./bert_cache/bert-base-uncased-local', freeze_bert=True)
   ```

4. **更多轮次**:
   ```bash
   -ncomm 150  # 训练更久
   ```

## ⚡ 性能优化

BERT 比 LSTM 慢很多 (110M vs 7M 参数):

- **LSTM**: ~30秒/轮
- **BERT**: ~3-5分钟/轮

如果太慢:
1. 减少客户端: `-nc 10`
2. 减少batch: `-B 8`
3. 使用更少epoch: `-E 1`

## 🆚 对比实验

建议同时跑 LSTM 和 BERT 对比:

### LSTM (快速baseline)
```bash
python server.py -mn lstm -vs 50000 -dsn squad -nc 20 -cf 0.5 -E 3 -B 64 -lr 0.002 -ncomm 100 -sf 10
```

### BERT (更强性能)
```bash
python server.py -mn bert -dsn squad -nc 20 -cf 0.5 -E 2 -B 16 -lr 0.00002 -ncomm 100 -sf 10
```

预期: BERT 的 EM 应该比 LSTM 高 15-25%
