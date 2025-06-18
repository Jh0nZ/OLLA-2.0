# text_encoder.py
from transformers import CLIPTokenizer, CLIPTextModel
import torch.nn as nn
import torch

class TextEncoder(nn.Module):
    def __init__(self, device='cpu'):
        super(TextEncoder, self).__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.output_dim = self.text_model.config.hidden_size  # 512

    def forward(self, text_list):
        """
        text_list: lista de strings
        returns: tensor de embeddings (batch_size, output_dim)
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt").to(self.text_model.device)
        outputs = self.text_model(**inputs)
        # Usamos el embedding del token [EOS] como representación del texto
        last_hidden_state = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        eos_embedding = last_hidden_state[torch.arange(last_hidden_state.size(0)), inputs['input_ids'].argmax(dim=1)]
        return eos_embedding  # (batch_size, output_dim)
