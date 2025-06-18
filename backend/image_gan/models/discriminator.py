import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, text_dim=512, img_channels=3):
        super().__init__()

        self.img_net = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, 2, 1),   # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),            # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc_text = nn.Linear(text_dim, 128 * 16 * 16)

        self.final = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=16),  # De (B,256,16,16) -> (B,1,1,1)
            nn.Sigmoid()
        )

    def forward(self, img, text_embedding):
        img_out = self.img_net(img)                      # (B, 128, 16, 16)
        text_out = self.fc_text(text_embedding)          # (B, 128*16*16)
        text_out = text_out.view(-1, 128, 16, 16)
        combined = torch.cat([img_out, text_out], dim=1) # (B, 256, 16, 16)
        validity = self.final(combined).view(-1)
        return validity
