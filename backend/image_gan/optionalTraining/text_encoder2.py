from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
    
    def forward(self, text_inputs):
        with torch.no_grad():
            output = self.bert(**text_inputs)
        return output.pooler_output  
