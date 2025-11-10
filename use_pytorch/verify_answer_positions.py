import sys
sys.path.append('..')

from get_dataset import GetDataset
from utils.nlp_transform import NLPTransform

def char_to_token_accurate(text, answer_text, char_start, nlp):
    """精确的字符到token位置映射"""
    tokens = nlp.basic_tokenizer(text)
    answer_tokens = nlp.basic_tokenizer(answer_text)
    
    # 方法1: 在tokens中直接查找答案token序列
    for i in range(len(tokens) - len(answer_tokens) + 1):
        if tokens[i:i+len(answer_tokens)] == answer_tokens:
            return i, i + len(answer_tokens) - 1
    
    # 方法2: 如果没找到，使用字符位置重建
    text_lower = text.lower()
    char_pos = 0
    token_char_map = []  # [(token_idx, char_start, char_end)]
    
    for idx, token in enumerate(tokens):
        # 在文本中找到这个token的位置
        token_start = text_lower.find(token, char_pos)
        if token_start >= 0:
            token_end = token_start + len(token)
            token_char_map.append((idx, token_start, token_end))
            char_pos = token_end
        else:
            # 处理找不到的情况（可能因为分词差异）
            token_char_map.append((idx, char_pos, char_pos))
    
    # 找到包含答案字符范围的token
    answer_char_end = char_start + len(answer_text)
    start_token = 0
    end_token = 0
    
    for idx, ts, te in token_char_map:
        # token的字符范围与答案重叠
        if ts <= char_start < te:
            start_token = idx
        if ts < answer_char_end <= te:
            end_token = idx
            break
    
    # 如果end_token没有更新，可能答案跨越多个token
    if end_token == 0 or end_token < start_token:
        end_token = min(start_token + len(answer_tokens) - 1, len(tokens) - 1)
    
    return start_token, end_token

# 加载数据
dataset = GetDataset('squad', '../data')
contexts, questions, answers = dataset.get_raw_data('train')

# 构建词汇表
print("=" * 80)
print("验证答案位置映射 - 新旧方法对比")
print("=" * 80)

nlp = NLPTransform(max_length=512)
all_texts = contexts[:1000] + questions[:1000]
nlp.build_vocab(all_texts, max_vocab_size=30000)

correct_old = 0
correct_new = 0
total = 0

for i in range(50):  # 测试前50个样本
    context = contexts[i]
    answer = answers[i]
    
    char_start = answer['start']
    char_end = answer['end']
    answer_text = answer['text']
    
    tokens = nlp.basic_tokenizer(context)
    answer_tokens = nlp.basic_tokenizer(answer_text)
    
    # 旧方法
    avg_token_len = len(context) / len(tokens) if len(tokens) > 0 else 1
    old_start = int(char_start / avg_token_len)
    old_end = int(char_end / avg_token_len)
    
    # 新方法
    new_start, new_end = char_to_token_accurate(context, answer_text, char_start, nlp)
    
    # 评估准确性
    if old_start < len(tokens) and old_end < len(tokens):
        old_pred = ' '.join(tokens[old_start:old_end+1])
        old_match = answer_text.lower() in old_pred.lower() or old_pred.lower() in answer_text.lower()
        if old_match:
            correct_old += 1
    
    if new_start < len(tokens) and new_end < len(tokens):
        new_pred = ' '.join(tokens[new_start:new_end+1])
        new_match = answer_text.lower() in new_pred.lower() or new_pred.lower() in answer_text.lower()
        if new_match:
            correct_new += 1
    
    total += 1
    
    # 打印前10个样本的详细信息
    if i < 10:
        print(f"\n{'='*80}")
        print(f"样本 {i}:")
        print(f"问题: {questions[i]}")
        print(f"标准答案: '{answer_text}'")
        
        print(f"\n旧方法 (平均长度):")
        print(f"  Token位置: {old_start} → {old_end}")
        if old_start < len(tokens) and old_end < len(tokens):
            old_pred = ' '.join(tokens[old_start:old_end+1])
            print(f"  预测答案: '{old_pred}'")
            print(f"  ✓ 正确" if old_match else "  ✗ 错误")
        
        print(f"\n新方法 (精确映射):")
        print(f"  Token位置: {new_start} → {new_end}")
        if new_start < len(tokens) and new_end < len(tokens):
            new_pred = ' '.join(tokens[new_start:new_end+1])
            print(f"  预测答案: '{new_pred}'")
            print(f"  ✓ 正确" if new_match else "  ✗ 错误")

print(f"\n{'='*80}")
print(f"准确率统计 (测试 {total} 个样本):")
print(f"  旧方法准确率: {correct_old}/{total} = {100*correct_old/total:.1f}%")
print(f"  新方法准确率: {correct_new}/{total} = {100*correct_new/total:.1f}%")
print(f"  提升: {correct_new - correct_old} 个样本")