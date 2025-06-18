import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_dim=768, z_dim=100, img_channels=3):
        super().__init__()
        self.text_dim = text_dim
        self.z_dim = z_dim

        # Proyección del embedding de texto a menor dimensión
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(True)
        )

        self.init_size = 8  # Tamaño base de la imagen (8x8)
        self.fc = nn.Sequential(
            nn.Linear(128 + z_dim, 256 * self.init_size * self.init_size),
            nn.ReLU(True)
        )

        # Bloques de Upsampling
        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )

        # Secuencia de bloques: cada uno incluye el text_feat como canal adicional
        self.conv_blocks = nn.Sequential(
            block(256 + 128, 128),   # 8x8 → 16x16
            block(128 + 128, 64),    # 16x16 → 32x32
            nn.Conv2d(64 + 128, img_channels, 3, stride=1, padding=1),  # 32x32 → 64x64
            nn.Tanh()
        )

    def forward(self, text_embedding, z):
        # Paso 1: Proyectar el texto a un espacio latente más pequeño
        text_feat = self.text_proj(text_embedding)  # (B, 128)

        # Paso 2: Concatenar texto + ruido para generar variedad
        x = torch.cat((text_feat, z), dim=1)  # (B, 128 + z_dim)
        x = self.fc(x)                        # (B, 256*8*8)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)  # (B, 256, 8, 8)
        # Paso 3: Primer inyección de texto
        text_feat_exp = text_feat.unsqueeze(2).unsqueeze(3)
        out = torch.cat((x, text_feat_exp.expand(-1, -1, x.size(2), x.size(3))), dim=1)

        # Paso 4: Samplear y volver a inyectar el texto
        for layer in self.conv_blocks:
            out = layer(out)
            # Si aún no es la imagen de salida seguimos inyectando texto
            if out.size(1) != 3:
                text_feat_exp = text_feat.unsqueeze(2).unsqueeze(3)
                text_feat_exp = text_feat_exp.expand(-1, -1, out.size(2), out.size(3))
                out = torch.cat((out, text_feat_exp), dim=1)

        return out
