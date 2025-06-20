from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class RecipeGenerator:
    def __init__(self, model_name='datificate/gpt2-small-spanish'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.eval()

    def generate(self, ingredientes, max_length=128, temperature=0.7):
        prompt = (
            f"<INPUTS> {ingredientes} <INPUTS_END>\n"
            f"<NAME>"
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
                num_return_sequences=3,
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
        # para entrenar
        # return text.strip()
        return self._extract_recipe_parts(text.strip())

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path):
        self.model = GPT2LMHeadModel.from_pretrained(path).to(self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(path)
        
        # En el método __init__ o load_model
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    def _extract_recipe_parts(self, text):
        """Extrae el nombre y procedimiento de la receta generada"""
        nombre = ""
        procedimiento = ""
        
        # Buscar el nombre de la receta
        if "<NAME>" in text and "<NAME_END>" in text:
            start_idx = text.find("<NAME>") + len("<NAME>")
            end_idx = text.find("<NAME_END>")
            nombre = text[start_idx:end_idx].strip()
        
        # Buscar las instrucciones
        if "<INSTRUCTIONS>" in text:
            start_idx = text.find("<INSTRUCTIONS>") + len("<INSTRUCTIONS>")
            if "<INSTRUCTIONS_END>" in text:
                end_idx = text.find("<INSTRUCTIONS_END>")
                procedimiento = text[start_idx:end_idx].strip()
            else:
                # Si no hay tag de cierre, tomar el resto del texto
                procedimiento = text[start_idx:].strip()
        
        return {
            "nombre": nombre if nombre else "Receta sin nombre",
            "procedimiento": procedimiento if procedimiento else "No se generó procedimiento",
            "texto_completo": text
        }