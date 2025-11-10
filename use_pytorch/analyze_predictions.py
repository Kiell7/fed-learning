import sys
sys.path.append('..')

import torch
import numpy as np
from clients import ClientsGroup
from Models import LSTM_QA

print("=" * 80)
print("分析模型预测行为")
print("=" * 80)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据和客户端组
print("加载数据集...")
clients_group = ClientsGroup(
    dataSetName='squad',
    isIID=True,
    numOfClients=100,
    dev=dev
)

# ✅ 从任意客户端的数据集中获取 nlp_transform
client0 = clients_group.clients_set['client0']
nlp_transform = client0.train_ds.nlp_transform
vocab_size = nlp_transform.vocab_size

print(f"词汇表大小: {vocab_size}")

# 加载训练好的模型
checkpoint_path = "./checkpoints/lstm_squad_nc100_comm100_20251106_212801/final_model.pth"
print(f"加载模型: {checkpoint_path}")

# ✅ 先加载checkpoint查看参数
checkpoint = torch.load(checkpoint_path)

# 检查checkpoint的结构
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    print(f"✅ 检测到完整的checkpoint，包含轮次={checkpoint.get('round')}, 准确率={checkpoint.get('accuracy'):.2f}%, loss={checkpoint.get('loss'):.4f}")
    
    # ✅ 从checkpoint中推断模型参数
    embedding_weight = checkpoint['model_state_dict']['embedding.weight']
    embedding_dim = embedding_weight.shape[1]  # 实际是100，不是128
    print(f"✅ 从checkpoint推断: embedding_dim={embedding_dim}")
    
    # 用正确的参数创建模型
    net = LSTM_QA(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=256)
    net.load_state_dict(checkpoint['model_state_dict'])
else:
    # 如果是直接保存的state_dict，尝试推断参数
    embedding_weight = checkpoint['embedding.weight']
    embedding_dim = embedding_weight.shape[1]
    print(f"✅ 从state_dict推断: embedding_dim={embedding_dim}")
    
    net = LSTM_QA(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=256)
    net.load_state_dict(checkpoint)

net = net.to(dev)
net.eval()

print("\n1. 分析模型预测分布")
print("-" * 80)

all_pred_starts = []
all_pred_ends = []
all_true_starts = []
all_true_ends = []

with torch.no_grad():
    batch_count = 0
    for batch_data in clients_group.test_data_loader:
        context, question, start_pos, end_pos = batch_data
        context = context.to(dev)
        question = question.to(dev)
        
        start_logits, end_logits = net(context, question)
        
        pred_start = torch.argmax(start_logits, dim=1).cpu().numpy()
        pred_end = torch.argmax(end_logits, dim=1).cpu().numpy()
        
        all_pred_starts.extend(pred_start.tolist())
        all_pred_ends.extend(pred_end.tolist())
        all_true_starts.extend(start_pos.numpy().tolist())
        all_true_ends.extend(end_pos.numpy().tolist())
        
        batch_count += 1
        if batch_count >= 10:  # 只看前1280个样本
            break

all_pred_starts = np.array(all_pred_starts)
all_pred_ends = np.array(all_pred_ends)
all_true_starts = np.array(all_true_starts)
all_true_ends = np.array(all_true_ends)

print(f"\n样本总数: {len(all_pred_starts)}")

print(f"\n预测的 start_pos 分布:")
print(f"  最小值: {all_pred_starts.min()}")
print(f"  最大值: {all_pred_starts.max()}")
print(f"  平均值: {all_pred_starts.mean():.1f}")
print(f"  中位数: {np.median(all_pred_starts):.1f}")
print(f"  标准差: {all_pred_starts.std():.1f}")

print(f"\n真实的 start_pos 分布:")
print(f"  最小值: {all_true_starts.min()}")
print(f"  最大值: {all_true_starts.max()}")
print(f"  平均值: {all_true_starts.mean():.1f}")
print(f"  中位数: {np.median(all_true_starts):.1f}")
print(f"  标准差: {all_true_starts.std():.1f}")

print(f"\n预测的答案长度:")
pred_lengths = all_pred_ends - all_pred_starts + 1
print(f"  平均长度: {pred_lengths.mean():.1f}")
print(f"  中位数: {np.median(pred_lengths):.1f}")
print(f"  最大长度: {pred_lengths.max()}")
print(f"  最小长度: {pred_lengths.min()}")

print(f"\n真实的答案长度:")
true_lengths = all_true_ends - all_true_starts + 1
print(f"  平均长度: {true_lengths.mean():.1f}")
print(f"  中位数: {np.median(true_lengths):.1f}")

# 检查是否所有预测都集中在某些位置
from collections import Counter
start_counter = Counter(all_pred_starts)
most_common_starts = start_counter.most_common(10)

print(f"\n⚠️ 检查模型是否总预测相同位置:")
print(f"出现最频繁的前10个 start_pos:")
for pos, count in most_common_starts:
    percentage = 100 * count / len(all_pred_starts)
    print(f"  位置 {pos}: {count}次 ({percentage:.1f}%)")

# 检查 end_pos
end_counter = Counter(all_pred_ends)
most_common_ends = end_counter.most_common(10)

print(f"\n出现最频繁的前10个 end_pos:")
for pos, count in most_common_ends:
    percentage = 100 * count / len(all_pred_ends)
    print(f"  位置 {pos}: {count}次 ({percentage:.1f}%)")

if most_common_starts[0][1] > len(all_pred_starts) * 0.5:
    print(f"\n❌❌❌ 严重问题：超过50%的预测都是位置 {most_common_starts[0][0]}！")
    print(f"   模型没有真正学习，只是记住了一个固定位置。")
elif most_common_starts[0][1] > len(all_pred_starts) * 0.2:
    print(f"\n⚠️ 问题：超过20%的预测都是位置 {most_common_starts[0][0]}")
    print(f"   模型的多样性不足。")
else:
    print(f"\n✅ 预测分布较为合理")

# 分析几个具体样本
print("\n" + "=" * 80)
print("2. 分析具体预测样例")
print("-" * 80)

for i in range(min(5, len(client0.train_ds))):
    sample = client0.train_ds[i]
    raw_data = client0.train_ds.data[i]
    
    if isinstance(raw_data, tuple) and len(raw_data) == 3:
        context_text, question_text, answer_dict = raw_data
        
        context = sample['context'].unsqueeze(0).to(dev)
        question = sample['question'].unsqueeze(0).to(dev)
        true_start = sample['start_pos'].item()
        true_end = sample['end_pos'].item()
        
        with torch.no_grad():
            start_logits, end_logits = net(context, question)
            pred_start = torch.argmax(start_logits, dim=1).item()
            pred_end = torch.argmax(end_logits, dim=1).item()
            
            # 获取前3个最可能的位置
            top3_starts = torch.topk(start_logits[0], 3)
            top3_ends = torch.topk(end_logits[0], 3)
        
        # 获取tokens
        tokens = nlp_transform.basic_tokenizer(context_text)
        
        print(f"\n{'='*60}")
        print(f"样本 {i}:")
        print(f"  问题: {question_text[:80]}...")
        print(f"  真实答案: '{answer_dict['text']}'")
        
        if pred_start < len(tokens) and pred_end < len(tokens) and pred_end >= pred_start:
            pred_answer = ' '.join(tokens[pred_start:pred_end+1])
            print(f"  预测答案: '{pred_answer}'")
        else:
            print(f"  预测答案: [无效位置 {pred_start}-{pred_end}]")
        
        print(f"  真实位置: {true_start}-{true_end} (长度={true_end-true_start+1})")
        print(f"  预测位置: {pred_start}-{pred_end} (长度={pred_end-pred_start+1})")
        print(f"  精确匹配: {'✅' if pred_start == true_start and pred_end == true_end else '❌'}")
        
        # 显示top3预测
        print(f"  Top3 start预测:")
        for idx, (score, pos) in enumerate(zip(top3_starts.values, top3_starts.indices)):
            print(f"    {idx+1}. 位置 {pos.item()}: {score.item():.3f}")
        
        print(f"  Top3 end预测:")
        for idx, (score, pos) in enumerate(zip(top3_ends.values, top3_ends.indices)):
            print(f"    {idx+1}. 位置 {pos.item()}: {score.item():.3f}")

print("\n" + "=" * 80)
print("3. 诊断结论")
print("=" * 80)

if most_common_starts[0][1] > len(all_pred_starts) * 0.3:
    print("\n❌ 主要问题：模型陷入了「局部最优」")
    print("   - 模型学会了总是预测某些固定位置来降低loss")
    print("   - 这不是真正的学习，只是找到了loss函数的漏洞")
    print("\n💡 解决方案:")
    print("   1. 增加客户端采样比例: -cf 0.3 或 0.5")
    print("   2. 减少客户端数量: -nc 50 或 20")
    print("   3. 增加学习率: -lr 0.002 或 0.003")
    print("   4. 添加 Dropout 防止过拟合固定模式")
    print("   5. ⚠️ 重新训练（当前模型已经学坏了）")
else:
    print("\n✅ 预测分布正常")
    print("   - 问题可能是数据不足或训练不充分")
    print("\n💡 解决方案:")
    print("   1. 大幅增加客户端采样: -cf 0.5")
    print("   2. 延长训练: -ncomm 200")

print("\n" + "=" * 80)
print("4. 推荐的训练命令")
print("=" * 80)
print("\n方案1（平衡）：")
print("python server.py -mn lstm -vs 50000 -dsn squad -nc 50 -cf 0.3 -E 5 -B 64 -lr 0.002 -ncomm 100 -sf 5")
print("\n方案2（激进，推荐）：")
print("python server.py -mn lstm -vs 50000 -dsn squad -nc 20 -cf 0.5 -E 3 -B 64 -lr 0.003 -ncomm 150 -sf 10")
print("\n方案3（极限，最快收敛）：")
print("python server.py -mn lstm -vs 50000 -dsn squad -nc 10 -cf 1.0 -E 2 -B 64 -lr 0.003 -ncomm 100 -sf 5")