import json
import os

squad_path = '/home/student4/njj/fed-learning-feature-dev/data/squad1.1'

# 检查文件
train_file = os.path.join(squad_path, 'train-v1.1.json')
dev_file = os.path.join(squad_path, 'dev-v1.1.json')

print(f"训练集存在: {os.path.exists(train_file)}")
print(f"验证集存在: {os.path.exists(dev_file)}")

if os.path.exists(train_file):
    with open(train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    print(f"\n数据集版本: {train_data.get('version', 'unknown')}")
    print(f"文章数量: {len(train_data['data'])}")
    
    # 查看第一个样本
    first_article = train_data['data'][0]
    print(f"\n第一篇文章标题: {first_article['title']}")
    
    first_paragraph = first_article['paragraphs'][0]
    print(f"\n第一个段落的上下文长度: {len(first_paragraph['context'])}")
    print(f"问题数量: {len(first_paragraph['qas'])}")
    
    first_qa = first_paragraph['qas'][0]
    print(f"\n第一个问答对:")
    print(f"问题ID: {first_qa['id']}")
    print(f"问题: {first_qa['question']}")
    print(f"答案: {first_qa['answers'][0]['text']}")
    print(f"答案起始位置: {first_qa['answers'][0]['answer_start']}")