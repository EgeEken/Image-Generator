# train_pix2pix.py
import os
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils
import mlflow

# -----------------------------
# Dataset
# -----------------------------
class Sketch2ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(256,256)):
        self.inputs = sorted(list(Path(input_dir).glob("*.jpg")))
        self.targets = sorted(list(Path(target_dir).glob("*.jpg")))
        self.image_size = image_size
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_img = cv2.imread(str(self.inputs[idx]), cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(str(self.targets[idx]))

        input_img = cv2.resize(input_img, self.image_size)
        target_img = cv2.resize(target_img, self.image_size)

        input_img = torch.tensor(input_img/255.0, dtype=torch.float32).unsqueeze(0)
        target_img = torch.tensor(target_img/255.0, dtype=torch.float32).permute(2,0,1)

        return input_img, target_img

# -----------------------------
# Generator (U-Net)
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2))
        self.down2 = nn.Sequential(nn.Conv2d(features, features*2, 4, 2, 1),
                                   nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2))
        self.down3 = nn.Sequential(nn.Conv2d(features*2, features*4, 4, 2, 1),
                                   nn.BatchNorm2d(features*4), nn.LeakyReLU(0.2))
        self.down4 = nn.Sequential(nn.Conv2d(features*4, features*8, 4, 2, 1),
                                   nn.BatchNorm2d(features*8), nn.LeakyReLU(0.2))
        # Decoder
        self.up1 = nn.Sequential(nn.ConvTranspose2d(features*8, features*4, 4, 2, 1),
                                 nn.BatchNorm2d(features*4), nn.ReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose2d(features*8, features*2, 4, 2, 1),
                                 nn.BatchNorm2d(features*2), nn.ReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose2d(features*4, features, 4, 2, 1),
                                 nn.BatchNorm2d(features), nn.ReLU())
        self.final = nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        u1 = self.up1(d4)
        u2 = self.up2(torch.cat([u1, d3], dim=1))
        u3 = self.up3(torch.cat([u2, d2], dim=1))
        out = self.final(torch.cat([u3, d1], dim=1))
        return self.tanh(out)

# -----------------------------
# PatchGAN Discriminator
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 4, 2, 1), nn.BatchNorm2d(features*2), nn.LeakyReLU(0.2),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.BatchNorm2d(features*4), nn.LeakyReLU(0.2),
            nn.Conv2d(features*4, 1, 4, 1, 1)  # output patch map
        )
    def forward(self, x):
        return self.model(x)

# -----------------------------
# Training Function
# -----------------------------
def train(
    input_dir, target_dir, epochs=10, batch_size=8, lr=2e-4, image_size=(256,256), checkpoint_dir="checkpoints"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    dataset = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    # Optimizers
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5,0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5,0.999))

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()

    # MLflow
    mlflow.start_run()
    mlflow.log_params({"epochs": epochs, "lr": lr, "batch_size": batch_size, "image_size": image_size})

    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            D.zero_grad()
            real_in = torch.cat([x, y], dim=1)
            pred_real = D(real_in)
            loss_D_real = criterion_adv(pred_real, torch.ones_like(pred_real))

            fake = G(x).detach()
            fake_in = torch.cat([x, fake], dim=1)
            pred_fake = D(fake_in)
            loss_D_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            G.zero_grad()
            fake = G(x)
            fake_in = torch.cat([x, fake], dim=1)
            pred_fake = D(fake_in)
            loss_G_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake))
            loss_G_l1 = criterion_l1(fake, y)
            loss_G = loss_G_adv + 100*loss_G_l1  # Pix2Pix weighting
            loss_G.backward()
            opt_G.step()

            if i % 50 == 0:
                print(f"[Epoch {epoch+1}] Batch {i} Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}")
                mlflow.log_metric("loss_G", loss_G.item(), step=epoch*len(loader)+i)
                mlflow.log_metric("loss_D", loss_D.item(), step=epoch*len(loader)+i)

        # ---------------------
        # Save checkpoint
        # ---------------------
        torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"G_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"D_epoch{epoch+1}.pth"))

        # Log sample outputs to MLflow
        G.eval()
        with torch.no_grad():
            sample_fake = G(x[:4])
            sample_grid = vutils.make_grid(sample_fake, nrow=2, normalize=True)
            mlflow.log_image(np.transpose((sample_grid.cpu().numpy()*255).astype(np.uint8), (1,2,0)), f"sample_epoch{epoch+1}.png")
        G.train()

    mlflow.end_run()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train(
        input_dir="../data_prepared/valid/inputs",
        target_dir="../data_prepared/valid/targets",
        epochs=1,
        batch_size=4,
        lr=2e-4,
        image_size=(256,256)
    )
