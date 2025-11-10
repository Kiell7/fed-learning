import sys
sys.path.append('..')

from get_dataset import GetDataset
from clients import ClientsGroup
import torch

print("=" * 80)
print("调试 SQuAD 数据集问题")
print("=" * 80)

# 加载数据
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 修正：参数名应该是 numOfClients
clients_group = ClientsGroup(
    dataSetName='squad', 
    isIID=True, 
    numOfClients=10, 
    dev=dev
)

# 获取第一个客户端的数据
client0 = clients_group.clients_set['client0']
dataset = client0.train_ds

print(f"\n数据集大小: {len(dataset)}")
print(f"数据集类型: {type(dataset)}")
print(f"is_squad: {dataset.is_squad}")

# 检查前10个样本
print("\n" + "=" * 80)
print("检查前10个样本的答案位置")
print("=" * 80)

for i in range(10):
    sample = dataset[i]
    
    context = sample['context']
    question = sample['question']
    start_pos = sample['start_pos'].item()
    end_pos = sample['end_pos'].item()
    
    print(f"\n样本 {i}:")
    print(f"  start_pos: {start_pos}")
    print(f"  end_pos: {end_pos}")
    print(f"  答案长度: {end_pos - start_pos + 1}")
    
    # 获取原始数据
    raw_data = dataset.data[i]
    if isinstance(raw_data, tuple) and len(raw_data) == 3:
        context_text, question_text, answer_dict = raw_data
        print(f"  原始答案: '{answer_dict['text']}'")
        print(f"  原始字符位置: {answer_dict['start']} - {answer_dict['end']}")
        
        # 检查token映射
        nlp = dataset.nlp_transform
        tokens = nlp.basic_tokenizer(context_text)
        
        print(f"  Context总token数: {len(tokens)}")
        
        if start_pos < len(tokens) and end_pos < len(tokens):
            predicted_answer = ' '.join(tokens[start_pos:end_pos+1])
            print(f"  预测答案: '{predicted_answer}'")
            print(f"  匹配: {answer_dict['text'].lower() in predicted_answer.lower()}")
        else:
            print(f"  ⚠️ 位置超出范围！")
            print(f"     token数={len(tokens)}, start={start_pos}, end={end_pos}")
            
            # 显示context的前后部分
            print(f"  Context前100字符: {context_text[:100]}")
            if len(context_text) > 100:
                print(f"  Context后100字符: ...{context_text[-100:]}")

# 统计位置分布
print("\n" + "=" * 80)
print("统计100个样本的位置分布")
print("=" * 80)

start_positions = []
end_positions = []
answer_lengths = []
token_counts = []

for i in range(min(100, len(dataset))):
    sample = dataset[i]
    start_pos = sample['start_pos'].item()
    end_pos = sample['end_pos'].item()
    
    start_positions.append(start_pos)
    end_positions.append(end_pos)
    answer_lengths.append(end_pos - start_pos + 1)
    
    # 获取实际token数
    raw_data = dataset.data[i]
    if isinstance(raw_data, tuple) and len(raw_data) == 3:
        context_text, _, _ = raw_data
        tokens = dataset.nlp_transform.basic_tokenizer(context_text)
        token_counts.append(len(tokens))

import numpy as np

print(f"start_pos 统计:")
print(f"  最小值: {min(start_positions)}")
print(f"  最大值: {max(start_positions)}")
print(f"  平均值: {np.mean(start_positions):.1f}")
print(f"  中位数: {np.median(start_positions):.1f}")

print(f"\nend_pos 统计:")
print(f"  最小值: {min(end_positions)}")
print(f"  最大值: {max(end_positions)}")
print(f"  平均值: {np.mean(end_positions):.1f}")
print(f"  中位数: {np.median(end_positions):.1f}")

print(f"\n答案长度统计:")
print(f"  最小值: {min(answer_lengths)}")
print(f"  最大值: {max(answer_lengths)}")
print(f"  平均值: {np.mean(answer_lengths):.1f}")
print(f"  中位数: {np.median(answer_lengths):.1f}")

print(f"\nContext token数统计:")
print(f"  最小值: {min(token_counts)}")
print(f"  最大值: {max(token_counts)}")
print(f"  平均值: {np.mean(token_counts):.1f}")
print(f"  中位数: {np.median(token_counts):.1f}")

# 检查是否有异常大的答案长度
abnormal = [i for i, l in enumerate(answer_lengths) if l > 20]
if abnormal:
    print(f"\n⚠️ 发现 {len(abnormal)} 个异常长的答案（>20 tokens）")
    print(f"异常样本索引: {abnormal[:10]}")
else:
    print(f"\n✅ 所有答案长度都在合理范围内（<=20 tokens）")

# 检查位置是否超出token范围
out_of_range = [i for i in range(len(token_counts)) 
                if start_positions[i] >= token_counts[i] or end_positions[i] >= token_counts[i]]

if out_of_range:
    print(f"\n⚠️⚠️⚠️ 发现 {len(out_of_range)} 个位置超出token范围的样本！")
    print(f"超出范围的样本索引: {out_of_range[:10]}")
    
    # 详细查看前3个超出范围的样本
    for idx in out_of_range[:3]:
        print(f"\n{'='*60}")
        print(f"详细分析样本 {idx}:")
        raw_data = dataset.data[idx]
        context_text, question_text, answer_dict = raw_data
        tokens = dataset.nlp_transform.basic_tokenizer(context_text)
        
        print(f"  原始答案: '{answer_dict['text']}'")
        print(f"  字符位置: {answer_dict['start']} - {answer_dict['end']}")
        print(f"  Token总数: {len(tokens)}")
        print(f"  映射后的位置: {start_positions[idx]} - {end_positions[idx]}")
        print(f"  Context长度: {len(context_text)} 字符")
        
        # 显示答案周围的内容
        ans_start = answer_dict['start']
        ans_end = answer_dict['end']
        snippet = context_text[max(0, ans_start-50):min(len(context_text), ans_end+50)]
        print(f"  答案上下文: ...{snippet}...")
else:
    print(f"\n✅ 所有位置都在token范围内")

print("\n" + "=" * 80)
print("调试完成")
print("=" * 80)