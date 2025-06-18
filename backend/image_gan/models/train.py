from datetime import datetime
import json
import torch
import torch.nn as nn
#import torch.nn.functional as F
import os
from text_encoder import TextEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
#from transformers import BertTokenizer, BertModel
from generator import Generator
from discriminator import Discriminator
import torchvision.transforms as transforms
from PIL import Image

class RecetasDataset(Dataset):
    def __init__(self, json_file, transform=None):
        self.root_dir = os.path.dirname(json_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data['recetas']
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        nombre = item['receta']
        img_rel_path = item['imagen']
        img_path = os.path.join(self.root_dir, "images", img_rel_path)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return nombre, image

# Hyperparámetros
batch_size = 16
lr = 0.0002
epochs = 50
text_dim = 512
z_dim = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset_path = os.path.join("..", "..", "datos", "dataset.json")
dataset = RecetasDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Directorio de salida
fecha_str = datetime.now().strftime("%Y%m%d")
output_dir = f"../outputs/generated_{fecha_str}"
os.makedirs(output_dir, exist_ok=True)

# Modelo de texto
text_encoder = TextEncoder(device=device).to(device)
text_encoder.eval()

# Redes
generator = Generator(text_dim=text_dim, z_dim=z_dim).to(device)
discriminator = Discriminator(text_dim=text_dim).to(device)

generator.train()
discriminator.train()

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


for epoch in range(epochs):
    
    for i, (nombres, imgs) in enumerate(dataloader):
        #print(f"Epoch {epoch+1} Batch {i} Nombres: {nombre}")  # <-- para ver qué datos entran en cada batch
        batch_size_curr = imgs.size(0)
        real = torch.ones(batch_size_curr, device=device)
        fake = torch.zeros(batch_size_curr, device=device)

        imgs = imgs.to(device)
        #input_ids = input_ids.to(device)
        #attention_mask = attention_mask.to(device)

        # Obtener embedding de texto
        with torch.no_grad():
                text_embeddings = text_encoder(nombres).to(device)
        
            
        # ------ (NUEVO) Generar embeddings incorrectos (rotados) ------
        with torch.no_grad():
            perm = torch.randperm(batch_size_curr)
            wrong_nombres = [nombres[j] for j in perm]
            wrong_embeddings = text_encoder(wrong_nombres).to(device)
        
        # --- Entrenar Discriminador ---
        noise = torch.randn(batch_size_curr, z_dim).to(device)
        fake_images = generator(text_embeddings, noise)

        pred_real = discriminator(imgs, text_embeddings)
        pred_fake = discriminator(fake_images.detach(), text_embeddings)
        pred_wrong = discriminator(imgs, wrong_embeddings)

        loss_real = criterion(pred_real, real)
        loss_fake = criterion(pred_fake, fake)
        loss_wrong = criterion(pred_wrong, fake)

        loss_D = (loss_real + loss_fake + loss_wrong) / 3

        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # --- Entrenar Generador ---
        pred_fake = discriminator(fake_images, text_embeddings)
        loss_G = criterion(pred_fake, real)  

        fake_images_wrong = generator(wrong_embeddings, noise)
        pred_wrong_fake = discriminator(fake_images_wrong, wrong_embeddings)
        loss_G_wrong = criterion(pred_wrong_fake, fake)  # debería ser rechazado
        
        loss_div = torch.mean((fake_images - fake_images_wrong) ** 2)

        loss_G_total = (loss_G + loss_G_wrong + 1.0 * loss_div) / 3.0

        optimizer_G.zero_grad()
        loss_G_total.backward()
        optimizer_G.step()

        #if i % 10 == 0:
        #    print(f"Epoch [{epoch+1}/{epochs}], Batch [{i}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

        if i % 10 == 0:
            generator.eval()
            with torch.no_grad():
                for idx in range(min(4, batch_size_curr)):  # guarda 4 imágenes del batch
                    text_emb = text_embeddings[idx].unsqueeze(0)
                    noise_sample = torch.randn(1, z_dim).to(device)
                    fake_img = generator(text_emb, noise_sample)
                    img_to_save = (fake_img + 1) / 2
                    save_image(img_to_save, f"{output_dir}/epoch{epoch+1}_batch{i}_img{idx}_{nombres[idx].replace(' ', '_')}.png")

            generator.train()

    print(f"Epoch [{epoch+1}/{epochs}] FINAL Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

torch.save(generator.state_dict(), f"../trains/generator/generator_{fecha_str}.pth")
torch.save(discriminator.state_dict(), f"../trains/discriminator/discriminator_{fecha_str}.pth")
