import re
from collections import Counter
import torch

class NLPTransform:
    def __init__(self, max_length=256):
        self.max_length = max_length
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
    
    def basic_tokenizer(self, text):
        """改进的分词器 - 更好地处理数字和特殊字符"""
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 转小写
        text = text.lower()
        
        # 关键改进：在标点符号和数字周围添加空格，但保留数字本身
        # 处理连字符：14-story -> 14 story, 1854–1855 -> 1854 1855
        text = re.sub(r'([0-9]+)[-–—]([0-9]+)', r'\1 \2', text)  # 数字-数字
        text = re.sub(r'([0-9]+)[-–—]([a-z]+)', r'\1 \2', text)  # 数字-字母（如14-story）
        
        # 在其他标点符号周围添加空格
        text = re.sub(r'([.,!?;:()\[\]{}"])', r' \1 ', text)
        
        # 移除除了字母、数字和空格之外的字符
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        
        # 分词并移除空token
        tokens = [t for t in text.split() if t]
        
        return tokens
    
    def build_vocab(self, texts, max_vocab_size=10000):
        """构建词汇表"""
        word_freq = Counter()
        
        for text in texts:
            tokens = self.basic_tokenizer(text)
            word_freq.update(tokens)
        
        # 保留最常见的词
        most_common = word_freq.most_common(max_vocab_size - 3)
        
        # 添加特殊标记
        self.word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2
        }
        self.idx2word = {
            0: '<PAD>',
            1: '<UNK>',
            2: '<START>'
        }
        
        # 添加常见词
        for idx, (word, _) in enumerate(most_common, start=3):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_size = len(self.word2idx)
        print(f"词汇表构建完成，包含 {self.vocab_size} 个词")
    
    def text_to_ids(self, text):
        """将文本转换为 token IDs"""
        tokens = self.basic_tokenizer(text)
        ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        # 填充或截断
        if len(ids) < self.max_length:
            ids += [self.word2idx['<PAD>']] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        return ids
    
    def imdb_transform(self, text):
        """IMDB 数据转换（保持兼容性）"""
        ids = self.text_to_ids(text)
        return torch.tensor(ids, dtype=torch.long)

