import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import glob

from models.generator import Generator
from models.text_encoder import TextEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parámetros
text_dim = 512
z_dim = 100
img_size = 64  # Debe coincidir con el tamaño que usaste en training

# Cargar encoder CLIP
text_encoder = TextEncoder(device=device).to(device)
text_encoder.eval()

# 🔍 Buscar el modelo generator más reciente
model_dir = "trains/generator"
model_files = sorted(glob.glob(os.path.join(model_dir, "generator_*.pth")), reverse=True)
if not model_files:
    raise FileNotFoundError("No se encontró ningún modelo guardado en trains/generator/")

latest_model_path = model_files[0]
print(f"🔄 Cargando modelo: {latest_model_path}")

# Carga el generator entrenado
generator = Generator(text_dim=text_dim, z_dim=z_dim).to(device)
generator.load_state_dict(torch.load(latest_model_path, map_location=device))
generator.eval()

#fixed_noise = torch.randn(1, z_dim).to(device)


def generate_image_from_text(text):
    # Obtener embedding desde CLIP
    with torch.no_grad():
        text_embedding = text_encoder([text]).to(device)  # shape: (1, 512)

    # Mostrar resumen del embedding
    emb_np = text_embedding.cpu().numpy()
    print(f"Embedding para '{text}':")
    print(f" - shape: {emb_np.shape}")
    print(f" - primeros 10 valores: {np.array2string(emb_np[0, :10], precision=4, floatmode='fixed')}")
    print(f" - media: {emb_np.mean():.4f}, std: {emb_np.std():.4f}\n")

    
    # Ruido aleatorio
    noise = torch.randn(1, z_dim).to(device)
    
    # Generar imagen fake
    with torch.no_grad():
        fake_img = generator(text_embedding, noise)
    
    # Post-proceso para PIL
    fake_img = fake_img.squeeze(0).cpu()
    fake_img = (fake_img + 1) / 2  # Normalizar a [0,1]
    fake_img = fake_img.clamp(0, 1)
    pil_img = transforms.ToPILImage()(fake_img)
    return pil_img

if __name__ == "__main__":
    print("Escribe el nombre del platillo para generar su imagen (o 'exit' para salir).")
    while True:
        nombre_platillo = input("Nombre: ").strip()
        if nombre_platillo.lower() == 'exit':
            break
        if len(nombre_platillo) == 0:
            print("Ingresa un texto válido.")
            continue
        img = generate_image_from_text(nombre_platillo)
        img.show()