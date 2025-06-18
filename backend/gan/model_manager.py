import json
from pathlib import Path
from .generator import RecipeGenerator
from .discriminator import DiscriminatorWrapper

MODELS_REGISTRY_PATH = Path(__file__).parent / "models" / "models_registry.json"

class ModelManager:
    @staticmethod
    def load_models_registry():
        if MODELS_REGISTRY_PATH.exists():
            with open(MODELS_REGISTRY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"models": []}
    
    @staticmethod
    def get_model_by_label(label: str):
        registry = ModelManager.load_models_registry()
        return next((m for m in registry["models"] if m["label"] == label), None)
    
    @staticmethod
    def get_model_by_id(model_id: int):
        registry = ModelManager.load_models_registry()
        return next((m for m in registry["models"] if m["id"] == model_id), None)
    
    @staticmethod
    def load_generator_and_discriminator(label: str = None, model_id: int = None):
        model_info = ModelManager.get_model_by_label(label) if label else ModelManager.get_model_by_id(model_id)
        if not model_info:
            return None, None
        
        try:
            base_path = Path(__file__).parent.parent.parent
            generator_path = base_path / model_info["generator_path"]
            discriminator_path = base_path / model_info["discriminator_path"]
            
            if not generator_path.exists() or not discriminator_path.exists():
                return None, None
            
            generator = RecipeGenerator()
            generator.load(str(generator_path))
            
            discriminator = DiscriminatorWrapper()
            discriminator.load(str(discriminator_path))
            
            return generator, discriminator
        except Exception:
            return None, None
    
    @staticmethod
    def list_models():
        registry = ModelManager.load_models_registry()
        return [{"label": m["label"], "description": m.get("description", "")} 
                for m in registry.get("models", [])]
    
    @staticmethod
    def delete_model(label: str = None, model_id: int = None):
        registry = ModelManager.load_models_registry()
        model_to_remove = (next((m for m in registry["models"] if m["label"] == label), None) if label 
                          else next((m for m in registry["models"] if m["id"] == model_id), None))
        
        if model_to_remove:
            registry["models"].remove(model_to_remove)
            MODELS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(MODELS_REGISTRY_PATH, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            return True
        return False

def load_model(label: str):
    return ModelManager.load_generator_and_discriminator(label=label)