import os
import json
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from .generator import RecipeGenerator
from .discriminator import DiscriminatorWrapper

# Ruta del dataset en formato JSON
DATASET_PATH = r"backend\datos\dataset.json"
# Ruta del registro de modelos
MODELS_REGISTRY_PATH = r"backend\gan\models\models_registry.json"

def load_models_registry():
    if os.path.exists(MODELS_REGISTRY_PATH):
        with open(MODELS_REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}

def save_models_registry(registry):
    os.makedirs(os.path.dirname(MODELS_REGISTRY_PATH), exist_ok=True)
    with open(MODELS_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

def register_model(generator_path, discriminator_path, label, description="", metrics=None):
    registry = load_models_registry()
    
    model_entry = {
        "id": len(registry["models"]) + 1,
        "label": label,
        "description": description,
        "timestamp": datetime.now().isoformat(),
        "generator_path": generator_path,
        "discriminator_path": discriminator_path,
        "metrics": metrics or {}
    }
    
    registry["models"].append(model_entry)
    save_models_registry(registry)
    
    return model_entry["id"]

def load_dataset(path: str = DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el dataset en {path}. Asegúrate de que la ruta sea correcta.")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data["recetas"]:
        # Obtener ingredientes como lista y convertir a string
        ingredientes_list = item.get("ingredientes", [])
        if isinstance(ingredientes_list, list):
            ingredientes = ", ".join(ingredientes_list)
        else:
            ingredientes = str(ingredientes_list)
        
        ingredientes = ingredientes.strip()
        
        # Obtener procedimiento
        procedimiento = item.get("procedimiento", "").strip()
        
        # Obtener nombre de la receta (opcional, para logging)
        nombre_receta = item.get("receta", "").strip()

        if ingredientes and procedimiento:
            processed.append({
                "receta": nombre_receta,
                "ingredientes": ingredientes,
                "procedimiento": procedimiento
            })
            
    if not processed:
        raise ValueError("El dataset está vacío o mal formateado. Revisa el archivo JSON.")

    return processed

def train(epochs: int = 50, lr_gen: float = 5e-5, lr_disc: float = 1e-5, batch_size: int = 4, model_label: str = None, model_description: str = ""):
    real_data = load_dataset()
    print(f"Dataset cargado: {len(real_data)} recetas")

    # Inicializamos modelos
    generator = RecipeGenerator()
    print("Generador cargado con éxito.")
    discriminator = DiscriminatorWrapper()
    print("Discriminador cargado con éxito.")

    # Optimizadores separados para cada modelo
    optimizer_gen = optim.Adam(generator.model.parameters(), lr=lr_gen)
    optimizer_disc = optim.Adam(discriminator.model.parameters(), lr=lr_disc)
    
    # Funciones de pérdida
    adversarial_loss = nn.BCELoss()

    # Variables para métricas
    epoch_gen_losses = []
    epoch_disc_losses = []

    # Entrenamiento
    for epoch in range(epochs):
        random.shuffle(real_data)
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        
        # Procesamos en batches
        for i in range(0, len(real_data), batch_size):
            batch = real_data[i:i+batch_size]
            
            generator.model.train()
            discriminator.model.eval()  # Congelamos el discriminador para este paso
            optimizer_gen.zero_grad()
            
            gen_loss = 0.0
            
            # Construir prompts para todo el batch
            prompts = []
            for sample in batch:
                nombre = sample["receta"]
                ingredientes = sample["ingredientes"]
                real_receta = sample["procedimiento"]
                prompt = (
                    f"<RECIPE_START> {nombre} <RECIPE_END>\n"
                    f"<INPUTS> {ingredientes} <INPUTS_END>\n"
                    f"<INSTRUCTIONS> {real_receta} <INSTRUCTIONS_END>"
                )
                prompts.append(prompt)

            # Tokenizar batch completo
            inputs = generator.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(generator.device)
                
            outputs = generator.model(**inputs, labels=inputs['input_ids'])
            lm_loss = outputs.loss

            # Generar texto para todo el batch
            generated_texts = [generator.generate(sample["ingredientes"]) for sample in batch]
            
            disc_scores = []
            for text in generated_texts:
                score = discriminator.predict(text)  # Asumimos que devuelve un escalar float
                disc_scores.append(score)
            
            disc_scores_tensor = torch.tensor(disc_scores, dtype=torch.float, device=discriminator.device).unsqueeze(1)
            real_labels = torch.ones_like(disc_scores_tensor)
            
            adversarial_gen_loss = adversarial_loss(disc_scores_tensor, real_labels)

            total_gen_batch_loss = lm_loss + 0.1 * adversarial_gen_loss
            total_gen_batch_loss.backward()
            optimizer_gen.step()

            total_gen_loss += total_gen_batch_loss.item()
            
            # ----- ENTRENAMIENTO DEL DISCRIMINADOR -----
            discriminator.model.train()
            optimizer_disc.zero_grad()
            
            disc_loss = 0.0
            
            for sample in batch:
                ingredientes = sample["ingredientes"]
                real_receta = sample["procedimiento"]
                nombre = sample["receta"]
                
                # Generar receta sintética
                generator.model.eval() # se pausa el generador para este paso
                with torch.no_grad():
                    fake_receta = generator.generate(ingredientes)
                
                respuesta_real = (
                    f"<RECIPE_START> {nombre} <RECIPE_END>\n"
                    f"<INSTRUCTIONS> {real_receta} <INSTRUCTIONS_END>"
                )
                inputs_real = discriminator.tokenizer(
                    respuesta_real, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                inputs_fake = discriminator.tokenizer(
                    fake_receta, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True,
                    max_length=128
                )
                
                # Mover a dispositivo
                inputs_real = {k: v.to(discriminator.device) for k, v in inputs_real.items()}
                inputs_fake = {k: v.to(discriminator.device) for k, v in inputs_fake.items()}
                
                # Predicciones del discriminador
                real_score = discriminator.model(**inputs_real)
                fake_score = discriminator.model(**inputs_fake)
                
                # Loss del discriminador
                real_labels = torch.ones_like(real_score)
                fake_labels = torch.zeros_like(fake_score)
                
                loss_real = adversarial_loss(real_score, real_labels)
                loss_fake = adversarial_loss(fake_score, fake_labels)
                
                disc_batch_loss = (loss_real + loss_fake) / 2
                disc_loss += disc_batch_loss
            
            # Backward pass del discriminador
            disc_loss = disc_loss / len(batch)
            disc_loss.backward()
            optimizer_disc.step()
            total_disc_loss += disc_loss.item()
        
        # Guardar métricas de la época
        epoch_gen_losses.append(total_gen_loss)
        epoch_disc_losses.append(total_disc_loss)
        
        # Mostrar progreso cada 5 épocas
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"  Pérdida Generador: {total_gen_loss:.4f}")
            print(f"  Pérdida Discriminador: {total_disc_loss:.4f}")
            
            # Generar ejemplo para verificar progreso
            sample_ingredients = "carne molida, tomate, cebolla, arroz, huevo, papa, pimienta"
            sample_recipe = generator.generate(sample_ingredients, max_length=128)
            print(f"  Ejemplo generado: {sample_recipe}...")
            print("-" * 50)

    # ----- GUARDADO DE MODELOS -----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = os.path.join("backend/gan/models/generator", timestamp)
    disc_dir = os.path.join("backend/gan/models/discriminator", timestamp)
    os.makedirs(gen_dir, exist_ok=True)
    os.makedirs(disc_dir, exist_ok=True)

    generator.save(gen_dir)
    discriminator.save(disc_dir)
    
    # Crear métricas para el registro
    metrics = {
        "epochs": epochs,
        "final_gen_loss": epoch_gen_losses[-1],
        "final_disc_loss": epoch_disc_losses[-1],
        "avg_gen_loss": sum(epoch_gen_losses) / len(epoch_gen_losses),
        "avg_disc_loss": sum(epoch_disc_losses) / len(epoch_disc_losses),
        "learning_rate_gen": lr_gen,
        "learning_rate_disc": lr_disc,
        "batch_size": batch_size,
        "dataset_size": len(real_data)
    }
    
    # Generar label si no se proporciona
    if not model_label:
        model_label = f"GAN_Model_{timestamp}"
    
    # Registrar modelo en JSON
    model_id = register_model(
        generator_path=gen_dir,
        discriminator_path=disc_dir,
        label=model_label,
        description=model_description,
        metrics=metrics
    )
    
    print(f"Modelos guardados y registrados:")
    print(f"  ID: {model_id}")
    print(f"  Label: {model_label}")
    print(f"  Generador: {gen_dir}")
    print(f"  Discriminador: {disc_dir}")
    print(f"  Registro: {MODELS_REGISTRY_PATH}")

if __name__ == "__main__":
    # Entrenamiento con más épocas y configuración mejorada
    train(
        epochs=50, 
        lr_gen=5e-5, 
        lr_disc=2e-5, 
        batch_size=2,
        model_label="Modelo_Recetas_Espanol_v1",
        model_description="Primer modelo entrenado con dataset de recetas en español, 50 épocas"
    )