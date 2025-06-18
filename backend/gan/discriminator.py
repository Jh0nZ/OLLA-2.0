import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class RecipeDiscriminator(nn.Module):
    def __init__(self, max_length=512):
        super(RecipeDiscriminator, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.max_length = max_length

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output.pooler_output
        return self.classifier(pooled_output)

class DiscriminatorWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = RecipeDiscriminator().to(self.device)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            score = self.model(**inputs).item()
        return score

    def save(self, path):
        torch.save(self.model.state_dict(), f"{path}/discriminator.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(f"{path}/discriminator.pt", map_location=self.device))
