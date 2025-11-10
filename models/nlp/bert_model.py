from transformers import BertModel, BertTokenizer
import torch.nn as nn

class BERTModel(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout=0.1):
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                          attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.fc(output)