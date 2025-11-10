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
            return i, i + len(answer_tokens) - 1, tokens, True
    
    # 方法2: 字符位置映射（作为后备）
    text_lower = text.lower()
    char_pos = 0
    token_char_map = []
    
    for idx, token in enumerate(tokens):
        token_start = text_lower.find(token, char_pos)
        if token_start >= 0:
            token_end = token_start + len(token)
            token_char_map.append((idx, token_start, token_end))
            char_pos = token_end
        else:
            token_char_map.append((idx, char_pos, char_pos))
    
    answer_char_end = char_start + len(answer_text)
    start_token = 0
    end_token = 0
    
    for idx, ts, te in token_char_map:
        if ts <= char_start < te:
            start_token = idx
        if ts < answer_char_end <= te:
            end_token = idx
            break
    
    if end_token == 0 or end_token < start_token:
        end_token = min(start_token + len(answer_tokens) - 1, len(tokens) - 1)
    
    return start_token, end_token, tokens, False

# 加载数据
print("加载 SQuAD 数据集...")
dataset = GetDataset('squad', '../data')
contexts, questions, answers = dataset.get_raw_data('train')

# 构建词汇表
print("构建词汇表...")
nlp = NLPTransform(max_length=512)
all_texts = contexts[:5000] + questions[:5000]
nlp.build_vocab(all_texts, max_vocab_size=50000)

print("\n" + "="*80)
print(f"在 1000 个样本上进行最终验证")
print("="*80)

method1_success = 0  # 直接匹配成功
method2_success = 0  # 字符映射成功
total = 0
failed_cases = []

for i in range(1000):
    context = contexts[i]
    answer = answers[i]
    
    char_start = answer['start']
    answer_text = answer['text']
    
    start_pos, end_pos, tokens, direct_match = char_to_token_accurate(
        context, answer_text, char_start, nlp
    )
    
    answer_tokens = nlp.basic_tokenizer(answer_text)
    
    if start_pos < len(tokens) and end_pos < len(tokens):
        pred_tokens = tokens[start_pos:end_pos+1]
        
        if pred_tokens == answer_tokens:
            if direct_match:
                method1_success += 1
            else:
                method2_success += 1
        else:
            failed_cases.append({
                'idx': i,
                'answer': answer_text,
                'answer_tokens': answer_tokens,
                'pred_tokens': pred_tokens,
                'context_snippet': context[max(0, char_start-50):char_start+len(answer_text)+50]
            })
    
    total += 1

print(f"\n结果:")
print(f"  总样本数: {total}")
print(f"  方法1成功 (直接token匹配): {method1_success} ({100*method1_success/total:.1f}%)")
print(f"  方法2成功 (字符位置映射): {method2_success} ({100*method2_success/total:.1f}%)")
print(f"  总成功率: {method1_success + method2_success}/{total} ({100*(method1_success+method2_success)/total:.1f}%)")
print(f"  失败: {len(failed_cases)} ({100*len(failed_cases)/total:.1f}%)")

if failed_cases:
    print(f"\n前5个失败案例:")
    for i, case in enumerate(failed_cases[:5]):
        print(f"\n{i+1}. 样本 {case['idx']}:")
        print(f"   答案: '{case['answer']}'")
        print(f"   答案tokens: {case['answer_tokens']}")
        print(f"   预测tokens: {case['pred_tokens']}")
        print(f"   上下文: ...{case['context_snippet']}...")
else:
    print("\n🎉 所有样本都成功匹配！")