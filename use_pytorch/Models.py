import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class LSTM_net(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, num_classes=2):
        super().__init__()
        print(f"初始化LSTM模型: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, (h_n, c_n) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_dim)
        # 使用最后一个时间步的输出
        out = self.fc(h_n[-1])  # h_n[-1]: (batch_size, hidden_dim)
        return out


class LSTM_QA(nn.Module):
    """用于 SQuAD 问答任务的 LSTM 模型"""
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super().__init__()
        print(f"初始化LSTM_QA模型: vocab_size={vocab_size}, embedding_dim={embedding_dim}, hidden_dim={hidden_dim}")
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 双向 LSTM 用于编码
        self.context_lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.3,
            bidirectional=True
        )
        
        self.question_lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            batch_first=True, 
            num_layers=2, 
            dropout=0.3,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        
        # 预测起始和结束位置
        self.start_fc = nn.Linear(hidden_dim * 4, 1)
        self.end_fc = nn.Linear(hidden_dim * 4, 1)
        
    def forward(self, context, question):
        """
        Args:
            context: (batch_size, context_len)
            question: (batch_size, question_len)
        Returns:
            start_logits: (batch_size, context_len)
            end_logits: (batch_size, context_len)
        """
        # 嵌入
        context_emb = self.embedding(context)  # (batch, context_len, emb_dim)
        question_emb = self.embedding(question)  # (batch, question_len, emb_dim)
        
        # 编码
        context_out, _ = self.context_lstm(context_emb)  # (batch, context_len, hidden*2)
        question_out, (h_n, _) = self.question_lstm(question_emb)  # (batch, question_len, hidden*2)
        
        # 使用问题的最后一个隐藏状态
        # h_n: (num_layers*2, batch, hidden) -> (batch, hidden*2)
        question_vec = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        # 扩展问题向量以匹配上下文长度
        question_vec_expanded = question_vec.unsqueeze(1).expand(-1, context_out.size(1), -1)
        
        # 连接上下文和问题表示
        combined = torch.cat([context_out, question_vec_expanded], dim=2)  # (batch, context_len, hidden*4)
        
        # 预测起始和结束位置
        start_logits = self.start_fc(combined).squeeze(-1)  # (batch, context_len)
        end_logits = self.end_fc(combined).squeeze(-1)  # (batch, context_len)
        
        return start_logits, end_logits


class BERT_QA(nn.Module):
    """用于 SQuAD 问答任务的 BERT 模型"""
    def __init__(self, model_path='./bert_cache/bert-base-uncased-local', freeze_bert=False):
        super().__init__()
        print(f"初始化BERT_QA模型: model_path={model_path}, freeze_bert={freeze_bert}")
        
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(model_path)
        
        # 是否冻结 BERT 参数 (只训练 QA 层)
        if freeze_bert:
            print("  冻结 BERT 参数，只训练 QA 输出层")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # QA 输出层
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.qa_outputs = nn.Linear(hidden_size, 2)  # 2 for start and end logits
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        print(f"  BERT hidden_size={hidden_size}, vocab_size={self.bert.config.vocab_size}")
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - 包含 [CLS] question [SEP] context [SEP]
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
            token_type_ids: (batch_size, seq_len) - 0 for question, 1 for context
        Returns:
            start_logits: (batch_size, seq_len)
            end_logits: (batch_size, seq_len)
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        sequence_output = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        sequence_output = self.dropout(sequence_output)
        
        # 预测起始和结束位置
        logits = self.qa_outputs(sequence_output)  # (batch_size, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)  # 各为 (batch_size, seq_len, 1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)  # (batch_size, seq_len)
        
        return start_logits, end_logits


class BERT_Classifier(nn.Module):
    """用于文本分类任务的 BERT 模型（如 IMDB 情感分类）"""
    def __init__(self, num_classes=2, model_path='./bert_cache/bert-base-uncased-local', freeze_bert=False):
        super().__init__()
        print(f"初始化BERT_Classifier模型: model_path={model_path}, num_classes={num_classes}, freeze_bert={freeze_bert}")
        
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(model_path)
        
        # 是否冻结 BERT 参数
        if freeze_bert:
            print("  冻结 BERT 参数，只训练分类层")
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 分类层
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        print(f"  BERT hidden_size={hidden_size}, vocab_size={self.bert.config.vocab_size}")
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        # BERT 编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用 [CLS] token 的输出做分类
        pooled_output = outputs.pooler_output  # (batch_size, hidden_size)
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)  # (batch_size, num_classes)
        
        return logits


