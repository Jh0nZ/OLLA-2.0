from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class RecipeGenerator:
    def __init__(self, model_name='datificate/gpt2-small-spanish'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.eval()

    def generate(self, ingredientes, max_length=128, temperature=0.9):
        prompt = (
            f"<INPUTS> {ingredientes} <INPUTS_END>\n"
            f"<RECIPE_START>\n"
            f"<INSTRUCTIONS>"
        )
        
        inputs = self.tokenizer.encode(
            prompt, 
            return_tensors='pt', 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_length=max_length,
                min_length=64,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                top_p=0.92,
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text.strip()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token