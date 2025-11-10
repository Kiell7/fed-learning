#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''

'''
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import sys,os
import json
import torch
from transformers import BertTokenizer

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改导入语句，直接从项目根目录导入
from utils.nlp_transform import NLPTransform
from utils import transform

class GetDataset():
    def __init__(self, dataset_name, root, use_bert=False):
        self.dataset_name = dataset_name
        self.root = root
        self.nlp_transform = None
        self.use_bert = use_bert  # 是否使用 BERT
        
        if dataset_name == 'imdb':
            self.nlp_transform = NLPTransform(max_length=256)

    def get_dataset(self):
        if self.dataset_name == 'mnist':
            train_dataset = datasets.MNIST(self.root, train=True, download=True)
            test_dataset = datasets.MNIST(self.root, train=False, download=True)
        elif self.dataset_name == 'cifar10':
            train_dataset = datasets.CIFAR10(self.root, train=True, download=True)
            test_dataset = datasets.CIFAR10(self.root, train=False, download=True)
        elif self.dataset_name == 'imdb':
            train_dataset, test_dataset = self._get_imdb_dataset()
        elif self.dataset_name == 'squad':
            train_dataset, test_dataset = self._get_squad_dataset()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        return train_dataset, test_dataset
    
    def get_raw_data(self, split='train'):
        """获取原始数据"""
        if self.dataset_name == 'imdb':
            imdb_path = os.path.join(self.root, 'aclImdb')
            split_dir = os.path.join(imdb_path, split)
            
            data = []
            labels = []
            
            for label in ['pos', 'neg']:
                label_dir = os.path.join(split_dir, label)
                label_val = 1 if label == 'pos' else 0
                
                for fname in os.listdir(label_dir):
                    if fname.endswith('.txt'):
                        with open(os.path.join(label_dir, fname), 'r', encoding='utf-8') as f:
                            text = f.read()
                            data.append(text)
                            labels.append(label_val)
            
            return data, labels
        elif self.dataset_name == 'squad':
            return self._get_squad_raw_data(split)
        else:
            raise ValueError(f"get_raw_data does not support {self.dataset_name}")
    
    def _get_squad_raw_data(self, split='train'):
        """获取 SQuAD 原始数据"""
        squad_path = os.path.join(self.root, 'squad1.1')
        file_name = 'train-v1.1.json' if split == 'train' else 'dev-v1.1.json'
        file_path = os.path.join(squad_path, file_name)
        
        contexts = []
        questions = []
        answers = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for article in data['data']:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    question = qa['question']
                    
                    # 获取答案（取第一个答案）
                    if qa['answers']:
                        answer_text = qa['answers'][0]['text']
                        answer_start = qa['answers'][0]['answer_start']
                        answer_end = answer_start + len(answer_text)
                        
                        contexts.append(context)
                        questions.append(question)
                        answers.append({
                            'text': answer_text,
                            'start': answer_start,
                            'end': answer_end
                        })
        
        print(f"加载了 {len(contexts)} 个 SQuAD {split} 样本")
        return contexts, questions, answers
    
    def _get_imdb_dataset(self):
        """加载IMDB数据集（返回已处理的数据集）"""
        # 获取原始数据
        train_data, train_labels = self.get_raw_data('train')
        test_data, test_labels = self.get_raw_data('test')
        
        # 构建词汇表
        print("正在构建词汇表...")
        self.nlp_transform.build_vocab(train_data, max_vocab_size=10000)
        print(f"词汇表大小: {self.nlp_transform.vocab_size}")
        
        # 创建数据集
        train_dataset = CustomDataset(train_data, train_labels, nlp_transform=self.nlp_transform)
        test_dataset = CustomDataset(test_data, test_labels, nlp_transform=self.nlp_transform)
        
        return train_dataset, test_dataset

    def _get_squad_dataset(self):
        """加载 SQuAD 数据集"""
        # 获取原始数据
        train_contexts, train_questions, train_answers = self._get_squad_raw_data('train')
        dev_contexts, dev_questions, dev_answers = self._get_squad_raw_data('dev')
        
        if self.use_bert:
            # 使用 BERT tokenizer
            print("使用 BERT tokenizer 处理 SQuAD 数据集...")
            train_dataset = BERTSQuADDataset(
                train_contexts,
                train_questions,
                train_answers,
                tokenizer_path='./bert_cache/bert-base-uncased-local',
                max_length=384
            )
            test_dataset = BERTSQuADDataset(
                dev_contexts,
                dev_questions,
                dev_answers,
                tokenizer_path='./bert_cache/bert-base-uncased-local',
                max_length=384
            )
        else:
            # 使用自定义 NLP transform
            print("正在为 SQuAD 构建词汇表...")
            self.nlp_transform = NLPTransform(max_length=512)
            
            all_texts = train_contexts + train_questions + dev_contexts + dev_questions
            self.nlp_transform.build_vocab(all_texts, max_vocab_size=50000)
            print(f"词汇表大小: {self.nlp_transform.vocab_size}")
            
            train_dataset = CustomSQuADDataset(
                train_contexts, 
                train_questions, 
                train_answers,
                nlp_transform=self.nlp_transform
            )
            test_dataset = CustomSQuADDataset(
                dev_contexts, 
                dev_questions, 
                dev_answers,
                nlp_transform=self.nlp_transform
            )
        
        return train_dataset, test_dataset


class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None, nlp_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.nlp_transform = nlp_transform
        
        if len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 3:
            self.is_squad = True
        else:
            self.is_squad = False

    def char_to_token_position(self, text, answer_text, char_start):
        """精确的字符到token位置映射 - 完全重写"""
        tokens = self.nlp_transform.basic_tokenizer(text)
        answer_tokens = self.nlp_transform.basic_tokenizer(answer_text)
        
        # 方法1: 直接在tokens中查找答案token序列（精确匹配）
        for i in range(len(tokens) - len(answer_tokens) + 1):
            if tokens[i:i+len(answer_tokens)] == answer_tokens:
                return i, i + len(answer_tokens) - 1
        
        # 方法2: 构建字符到token的映射表
        text_lower = text.lower()
        current_char_pos = 0
        char_to_token_map = {}  # 字符位置 -> token索引
        
        for token_idx, token in enumerate(tokens):
            # 在文本中找到这个token
            token_start = text_lower.find(token, current_char_pos)
            
            if token_start >= 0:
                # 记录这个token覆盖的每个字符位置
                for char_pos in range(token_start, token_start + len(token)):
                    char_to_token_map[char_pos] = token_idx
                current_char_pos = token_start + len(token)
        
        # 找到答案的起始和结束字符对应的token
        answer_char_end = char_start + len(answer_text)
        
        # 查找答案起始字符对应的token
        start_token = char_to_token_map.get(char_start, 0)
        
        # 查找答案结束字符对应的token（向前查找最后一个有效字符）
        end_token = start_token
        for char_pos in range(answer_char_end - 1, char_start - 1, -1):
            if char_pos in char_to_token_map:
                end_token = char_to_token_map[char_pos]
                break
        
        # 确保end >= start
        if end_token < start_token:
            end_token = start_token
        
        # 边界检查
        start_token = max(0, min(start_token, len(tokens) - 1))
        end_token = max(start_token, min(end_token, len(tokens) - 1))
        
        return start_token, end_token

    def __getitem__(self, idx):
        if self.is_squad:
            context, question, answer = self.data[idx]
            
            # 使用改进的位置映射
            start_pos, end_pos = self.char_to_token_position(
                context,
                answer['text'],
                answer['start']
            )
            
            # 转换为 token IDs（会截断到512）
            context_ids = self.nlp_transform.text_to_ids(context)
            question_ids = self.nlp_transform.text_to_ids(question)
            
            # 如果答案位置超过512，则截断
            start_pos = min(start_pos, 511)
            end_pos = min(end_pos, 511)
            
            # 确保 end >= start
            if end_pos < start_pos:
                end_pos = start_pos
            
            return {
                'context': torch.tensor(context_ids, dtype=torch.long),
                'question': torch.tensor(question_ids, dtype=torch.long),
                'start_pos': torch.tensor(start_pos, dtype=torch.long),
                'end_pos': torch.tensor(end_pos, dtype=torch.long)
            }
        else:
            # MNIST/CIFAR10/IMDB 数据处理
            x = self.data[idx]
            y = self.labels[idx]

            if isinstance(x, str):
                if self.nlp_transform is None:
                    raise ValueError("文本数据需要提供 nlp_transform")
                x = self.nlp_transform.imdb_transform(x)
                y = torch.tensor(y, dtype=torch.long)
            elif self.transform is not None:
                x = self.transform(x)

            return x, y

    def __len__(self):
        return len(self.data)

class CustomSQuADDataset(Dataset):
    def __init__(self, contexts, questions, answers, nlp_transform):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.nlp_transform = nlp_transform
    
    def char_to_token_position(self, text, answer_text, char_start):
        """精确的字符到token位置映射 - 与CustomDataset保持一致"""
        tokens = self.nlp_transform.basic_tokenizer(text)
        answer_tokens = self.nlp_transform.basic_tokenizer(answer_text)
        
        # 方法1: 直接在tokens中查找答案token序列
        for i in range(len(tokens) - len(answer_tokens) + 1):
            if tokens[i:i+len(answer_tokens)] == answer_tokens:
                return i, i + len(answer_tokens) - 1
        
        # 方法2: 构建字符到token的映射表
        text_lower = text.lower()
        current_char_pos = 0
        char_to_token_map = {}
        
        for token_idx, token in enumerate(tokens):
            token_start = text_lower.find(token, current_char_pos)
            if token_start >= 0:
                for char_pos in range(token_start, token_start + len(token)):
                    char_to_token_map[char_pos] = token_idx
                current_char_pos = token_start + len(token)
        
        answer_char_end = char_start + len(answer_text)
        start_token = char_to_token_map.get(char_start, 0)
        
        end_token = start_token
        for char_pos in range(answer_char_end - 1, char_start - 1, -1):
            if char_pos in char_to_token_map:
                end_token = char_to_token_map[char_pos]
                break
        
        if end_token < start_token:
            end_token = start_token
        
        start_token = max(0, min(start_token, len(tokens) - 1))
        end_token = max(start_token, min(end_token, len(tokens) - 1))
        
        return start_token, end_token
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # ✅ 使用正确的位置映射方法
        start_pos, end_pos = self.char_to_token_position(
            context,
            answer['text'],
            answer['start']
        )
        
        # 转换为 token IDs
        context_ids = self.nlp_transform.text_to_ids(context)
        question_ids = self.nlp_transform.text_to_ids(question)
        
        # 如果答案位置超过512，则截断
        start_pos = min(start_pos, 511)
        end_pos = min(end_pos, 511)
        
        # 确保 end >= start
        if end_pos < start_pos:
            end_pos = start_pos
        
        return {
            'context': torch.tensor(context_ids, dtype=torch.long),
            'question': torch.tensor(question_ids, dtype=torch.long),
            'start_pos': torch.tensor(start_pos, dtype=torch.long),
            'end_pos': torch.tensor(end_pos, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.contexts)


class BERTSQuADDataset(Dataset):
    """用于 BERT 的 SQuAD 数据集"""
    def __init__(self, contexts, questions, answers, tokenizer_path='./bert_cache/bert-base-uncased-local', max_length=384):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.max_length = max_length
        
        # 加载 BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print(f"  加载 BERT tokenizer: vocab_size={self.tokenizer.vocab_size}, max_length={max_length}")
    
    def __getitem__(self, idx):
        context = self.contexts[idx]
        question = self.questions[idx]
        answer = self.answers[idx]
        
        # BERT 输入格式: [CLS] question [SEP] context [SEP]
        # 先对 question 和 context 分别编码，获取答案位置
        question_tokens = self.tokenizer.tokenize(question)
        context_tokens = self.tokenizer.tokenize(context)
        
        # 计算最大context长度 (留出空间给 [CLS], [SEP], question, [SEP])
        max_context_len = self.max_length - len(question_tokens) - 3
        
        # 如果context太长，截断
        if len(context_tokens) > max_context_len:
            context_tokens = context_tokens[:max_context_len]
        
        # 找到答案在tokenized context中的位置
        answer_text = answer['text']
        answer_tokens = self.tokenizer.tokenize(answer_text)
        
        # 在context_tokens中查找answer_tokens
        start_pos = 0
        end_pos = 0
        found = False
        
        for i in range(len(context_tokens) - len(answer_tokens) + 1):
            if context_tokens[i:i+len(answer_tokens)] == answer_tokens:
                start_pos = i
                end_pos = i + len(answer_tokens) - 1
                found = True
                break
        
        # 如果没找到，使用更宽松的匹配
        if not found:
            # 尝试部分匹配
            for i in range(len(context_tokens)):
                match_count = 0
                for j, ans_token in enumerate(answer_tokens):
                    if i + j < len(context_tokens) and context_tokens[i + j] == ans_token:
                        match_count += 1
                    else:
                        break
                if match_count > len(answer_tokens) * 0.5:  # 至少匹配50%
                    start_pos = i
                    end_pos = min(i + len(answer_tokens) - 1, len(context_tokens) - 1)
                    break
        
        # 调整位置：在完整序列中的实际位置
        # [CLS] question [SEP] context [SEP]
        # 0     1..q     q+1   q+2..c   c+1
        question_len = len(question_tokens)
        start_pos_in_seq = start_pos + question_len + 2  # +2 for [CLS] and [SEP]
        end_pos_in_seq = end_pos + question_len + 2
        
        # 构建输入序列
        tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
        
        # 转换为 IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Token type IDs: 0 for question, 1 for context
        token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(context_tokens) + 1)
        
        # Attention mask: 1 for real tokens
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length
        
        # 截断
        input_ids = input_ids[:self.max_length]
        token_type_ids = token_type_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        
        # 确保位置不超过序列长度
        start_pos_in_seq = min(start_pos_in_seq, self.max_length - 1)
        end_pos_in_seq = min(end_pos_in_seq, self.max_length - 1)
        
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'start_pos': torch.tensor(start_pos_in_seq, dtype=torch.long),
            'end_pos': torch.tensor(end_pos_in_seq, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.contexts)


class BERTIMDBDataset(Dataset):
    """BERT 用于 IMDB 情感分类的数据集"""
    def __init__(self, texts, labels, tokenizer_path='./bert_cache/bert-base-uncased-local', max_length=512):
        """
        Args:
            texts: 文本列表
            labels: 标签列表 (0 或 1)
            tokenizer_path: BERT tokenizer 路径
            max_length: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        print(f"  加载 BERT tokenizer: vocab_size={self.tokenizer.vocab_size}, max_length={max_length}")
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # (max_length,)
            'attention_mask': encoding['attention_mask'].squeeze(0),  # (max_length,)
            'token_type_ids': encoding['token_type_ids'].squeeze(0),  # (max_length,)
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.texts)

