import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, text_dim=768, img_channels=3):
        super().__init__()

        # Convs que reducen la imagen; para 64x64 -> 16x16, para 256x256 -> 64x64
        self.img_net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),   # Halve spatial size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),            # Halve spatial size
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Proyecta texto a vector pequeño, no al tamaño espacial fijo
        self.fc_text = nn.Linear(text_dim, 128 * 4 * 4)  # vector pequeño
        self.final = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=4),  # kernel igual a spatial size después de conv_text
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        img_out = self.img_net(img)        # (B, 128, H_img, W_img)
        b, _, h, w = img_out.size()

        # Texto proyectado a tamaño 4x4 fijo
        text_out = self.fc_text(text_embedding)     # (B, 128*4*4)
        text_out = text_out.view(-1, 128, 4, 4)

        # Interpolamos el texto al tamaño espacial de img_out para concatenar
        text_out = F.interpolate(text_out, size=(h, w), mode='bilinear', align_corners=False)

        combined = torch.cat([img_out, text_out], dim=1)  # (B, 256, H_img, W_img)

        # Aquí asumimos que kernel_size de final conv es igual a (h,w)
        # Si quieres un tamaño variable, adapta kernel_size dinámicamente o usa pooling

        # Si kernel_size no coincide con (h,w), mejor hacer pooling para reducir
        # Ejemplo para kernel fijo 4x4 y entradas 16x16: podemos hacer pooling primero

        # Para simplificar, hagamos un AdaptiveAvgPool antes de final conv:
        pooled = F.adaptive_avg_pool2d(combined, (4, 4))  # (B, 256, 4, 4)

        validity = self.final(pooled).view(-1)

        return validity
