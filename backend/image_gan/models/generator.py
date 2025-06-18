import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, text_dim=512, z_dim=100, img_channels=3):
        super().__init__()
        self.text_fc = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(True)
        )


        self.fc = nn.Linear(128 + z_dim, 256 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)  # 8x8 → 16x16
        self.bn1 = nn.BatchNorm2d(128)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)   # 16x16 → 32x32
        self.bn2 = nn.BatchNorm2d(64)

        self.deconv3 = nn.ConvTranspose2d(64, img_channels, 4, 2, 1)  # 32x32 → 64x64

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()

    def forward(self, text_embedding, noise):
        text_feat = self.text_fc(text_embedding)              # (B, 128)
        x = torch.cat((text_feat, noise), dim=1)              # (B, 228)
        x = self.fc(x).view(-1, 256, 8, 8)

        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.tanh(self.deconv3(x))
        return x
