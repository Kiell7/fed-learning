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
            return i, i + len(answer_tokens) - 1, tokens
    
    # 方法2: 使用字符位置映射
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
    
    return start_token, end_token, tokens

# 加载数据
dataset = GetDataset('squad', '../data')
contexts, questions, answers = dataset.get_raw_data('train')

nlp = NLPTransform(max_length=512)
all_texts = contexts[:1000] + questions[:1000]
nlp.build_vocab(all_texts, max_vocab_size=30000)

print("=" * 80)
print("分析映射失败的样本")
print("=" * 80)

failed_cases = []

for i in range(100):  # 测试前100个样本
    context = contexts[i]
    answer = answers[i]
    
    char_start = answer['start']
    answer_text = answer['text']
    
    # 新方法
    new_start, new_end, tokens = char_to_token_accurate(context, answer_text, char_start, nlp)
    answer_tokens = nlp.basic_tokenizer(answer_text)
    
    if new_start < len(tokens) and new_end < len(tokens):
        new_pred = ' '.join(tokens[new_start:new_end+1])
        # 严格匹配：预测的token序列必须与答案token序列完全一致
        pred_tokens = tokens[new_start:new_end+1]
        exact_match = (pred_tokens == answer_tokens)
        
        if not exact_match:
            failed_cases.append({
                'idx': i,
                'question': questions[i],
                'answer': answer_text,
                'answer_tokens': answer_tokens,
                'predicted': new_pred,
                'pred_tokens': pred_tokens,
                'context_snippet': context[max(0, char_start-50):char_start+len(answer_text)+50]
            })

print(f"\n找到 {len(failed_cases)} 个失败案例")
print("\n详细分析前10个失败案例：\n")

for i, case in enumerate(failed_cases[:10]):
    print(f"{'='*80}")
    print(f"失败案例 {i+1} (样本 {case['idx']}):")
    print(f"问题: {case['question']}")
    print(f"\n标准答案: '{case['answer']}'")
    print(f"答案tokens: {case['answer_tokens']}")
    print(f"\n预测答案: '{case['predicted']}'")
    print(f"预测tokens: {case['pred_tokens']}")
    print(f"\n上下文片段:")
    print(f"...{case['context_snippet']}...")
    print(f"\n失败原因分析:")
    
    # 分析失败原因
    if len(case['pred_tokens']) != len(case['answer_tokens']):
        print(f"  → Token数量不匹配: {len(case['pred_tokens'])} vs {len(case['answer_tokens'])}")
    
    for j, (pred_tok, ans_tok) in enumerate(zip(case['pred_tokens'], case['answer_tokens'])):
        if pred_tok != ans_tok:
            print(f"  → Token {j} 不匹配: '{pred_tok}' vs '{ans_tok}'")

print(f"\n{'='*80}")
print("失败原因统计:")

# 统计失败原因
reasons = {
    'punctuation': 0,  # 标点符号问题
    'case': 0,         # 大小写问题（不应该有，因为都转小写了）
    'tokenization': 0, # 分词差异
    'position': 0      # 位置定位错误
}

for case in failed_cases:
    answer_clean = case['answer'].lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    pred_clean = case['predicted'].lower().replace('.', '').replace(',', '').replace('?', '').replace('!', '')
    
    if answer_clean == pred_clean:
        reasons['punctuation'] += 1
    elif case['answer'].lower() == case['predicted'].lower():
        reasons['case'] += 1
    elif set(case['answer_tokens']) == set(case['pred_tokens']):
        reasons['position'] += 1
    else:
        reasons['tokenization'] += 1

print(f"  标点符号问题: {reasons['punctuation']}")
print(f"  分词差异: {reasons['tokenization']}")
print(f"  位置定位错误: {reasons['position']}")
print(f"  其他: {len(failed_cases) - sum(reasons.values())}")