import sys
sys.path.append('..')

from utils.nlp_transform import NLPTransform

nlp = NLPTransform()

# 测试案例
test_cases = [
    "in the 1854–1855 academic year",
    "the 14-story Theodore M. Hesburgh Library",
    "Super Bowl 50 was held in 2015-2016",
    "The price is $14.99",
    "He scored 3-2 in the game"
]

print("=" * 80)
print("测试改进的 Tokenizer")
print("=" * 80)

for text in test_cases:
    tokens = nlp.basic_tokenizer(text)
    print(f"\n原文: {text}")
    print(f"分词: {tokens}")

# 测试具体的失败案例
print("\n" + "=" * 80)
print("测试失败案例")
print("=" * 80)

context1 = "ees, in the form of a Master of Arts (MA), in the 1854–1855 academic year. The program expanded to inclu"
answer1 = "1854"
print(f"\n案例1:")
print(f"上下文: ...{context1}...")
print(f"答案: '{answer1}'")
tokens1 = nlp.basic_tokenizer(context1)
answer_tokens1 = nlp.basic_tokenizer(answer1)
print(f"上下文tokens: {tokens1}")
print(f"答案tokens: {answer_tokens1}")

# 查找答案位置
if answer_tokens1[0] in tokens1:
    pos = tokens1.index(answer_tokens1[0])
    print(f"找到位置: {pos}")
    print(f"提取的token: '{tokens1[pos]}'")
else:
    print("❌ 未找到答案token")

context2 = "he colleges and schools. The main building is the 14-story Theodore M. Hesburgh Library, completed in"
answer2 = "14"
print(f"\n案例2:")
print(f"上下文: ...{context2}...")
print(f"答案: '{answer2}'")
tokens2 = nlp.basic_tokenizer(context2)
answer_tokens2 = nlp.basic_tokenizer(answer2)
print(f"上下文tokens: {tokens2}")
print(f"答案tokens: {answer_tokens2}")

if answer_tokens2[0] in tokens2:
    pos = tokens2.index(answer_tokens2[0])
    print(f"找到位置: {pos}")
    print(f"提取的token: '{tokens2[pos]}'")
else:
    print("❌ 未找到答案token")