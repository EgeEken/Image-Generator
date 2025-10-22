# train.py
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

print("Imports complete.")
print(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

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
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=4):
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        # Encoder
        prev_channels = in_channels
        for i in range(depth):
            out_ch = base_features * (2**i)
            layer = nn.Sequential(
                nn.Conv2d(prev_channels, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2)
            )
            self.encoders.append(layer)
            prev_channels = out_ch

        # Decoder
        for i in reversed(range(depth-1)):
            in_ch = prev_channels
            out_ch = base_features * (2**i)
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ))
            prev_channels = out_ch * 2  # because of skip connections

        self.final = nn.ConvTranspose2d(prev_channels//2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc_features = []
        out = x
        for layer in self.encoders:
            out = layer(out)
            enc_features.append(out)
        for i, layer in enumerate(self.decoders):
            skip = enc_features[-(i+2)]
            out = layer(out)
            out = torch.cat([out, skip], dim=1)
        out = self.final(out)
        return self.tanh(out)

# -----------------------------
# PatchGAN Discriminator
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_features=64, depth=3):
        super().__init__()
        self.model = nn.ModuleList()
        prev_channels = in_channels
        for i in range(depth-1):
            out_ch = base_features * (2**i)
            self.model.append(nn.Sequential(
                nn.Conv2d(prev_channels, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch) if i > 0 else nn.Identity(),
                nn.LeakyReLU(0.2)
            ))
            prev_channels = out_ch
        # final conv to patch
        self.model.append(nn.Conv2d(prev_channels, 1, 4, 1, 1))

    def forward(self, x):
        out = x
        for layer in self.model:
            out = layer(out)
        return out

# -----------------------------
# Training Function
# -----------------------------
def train(
    input_dir, target_dir,
    epochs=5,
    batch_size=8,
    image_size=(256,256),
    # Generator / Discriminator params
    gen_base_features=64, gen_depth=4, gen_lr=2e-4,
    disc_base_features=64, disc_depth=3, disc_lr=2e-4,
    # Loss weights
    adv_weight=1.0, l1_weight=100.0,
    # Label smoothing
    real_label_weight=0.9, fake_label_weight=0.1,
    checkpoint_dir="checkpoints",
    sample_count=4
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset
    dataset = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    G = UNetGenerator(base_features=gen_base_features, depth=gen_depth).to(device)
    D = PatchDiscriminator(base_features=disc_base_features, depth=disc_depth).to(device)

    # Optimizers
    opt_G = torch.optim.AdamW(G.parameters(), lr=gen_lr, betas=(0.5,0.999))
    opt_D = torch.optim.AdamW(D.parameters(), lr=disc_lr, betas=(0.5,0.999))

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()

    # MLflow
    mlflow.start_run()
    mlflow.log_params({
        "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
        "gen_base_features": gen_base_features, "gen_depth": gen_depth, "gen_lr": gen_lr,
        "disc_base_features": disc_base_features, "disc_depth": disc_depth, "disc_lr": disc_lr,
        "adv_weight": adv_weight, "l1_weight": l1_weight,
        "real_label_weight": real_label_weight, "fake_label_weight": fake_label_weight,
        "sample_count": sample_count
    })

    for epoch in range(epochs):
        for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x, y = x.to(device), y.to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            D.zero_grad()
            real_in = torch.cat([x, y], dim=1)
            pred_real = D(real_in)
            loss_D_real = criterion_adv(pred_real, torch.ones_like(pred_real)*real_label_weight)

            fake = G(x).detach()
            fake_in = torch.cat([x, fake], dim=1)
            pred_fake = D(fake_in)
            loss_D_fake = criterion_adv(pred_fake, torch.ones_like(pred_fake)*fake_label_weight)

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
            loss_G = adv_weight*loss_G_adv + l1_weight*loss_G_l1
            loss_G.backward()
            opt_G.step()

            # MLflow metrics every 50 steps
            if i % 50 == 0:
                mlflow.log_metric("loss_G", loss_G.item(), step=epoch*len(loader)+i)
                mlflow.log_metric("loss_D", loss_D.item(), step=epoch*len(loader)+i)

            # Print 10 times per epoch
            if i % max(1, len(loader)//10) == 0:
                print(f"[Epoch {epoch+1}] Step {i}/{len(loader)} Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}")
                # save one sample input/output/expected output trio to artifacts/epoch_{epoch+1}/step_{i}/"input.png" etc.
                sample_dir = f"artifacts/epoch_{epoch+1}/step_{i}"
                os.makedirs(sample_dir, exist_ok=True)
                cv2.imwrite(os.path.join(sample_dir, "input.png"), (x[0,0].cpu().numpy()*255).astype(np.uint8))
                cv2.imwrite(os.path.join(sample_dir, "output.png"), (fake[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                cv2.imwrite(os.path.join(sample_dir, "expected_output.png"), (y[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))


        # ---------------------
        # Save checkpoint
        # ---------------------
        torch.save(G.state_dict(), os.path.join(checkpoint_dir, f"G_epoch{epoch+1}.pth"))
        torch.save(D.state_dict(), os.path.join(checkpoint_dir, f"D_epoch{epoch+1}.pth"))

        # Log sample outputs to MLflow
        G.eval()
        with torch.no_grad():
            sample_fake = G(x[:sample_count])
            os.makedirs(f"artifacts/epoch{epoch+1}", exist_ok=True)
            for j in range(sample_count):
                sample_dir = f"artifacts/epoch{epoch+1}/sample{j+1}"
                os.makedirs(sample_dir, exist_ok=True)
                cv2.imwrite(os.path.join(sample_dir, "input.png"), (x[j,0].cpu().numpy()*255).astype(np.uint8))
                cv2.imwrite(os.path.join(sample_dir, "output.png"), (sample_fake[j].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
                cv2.imwrite(os.path.join(sample_dir, "expected_output.png"), (y[j].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        G.train()

    mlflow.end_run()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    train(
        input_dir="workspace/data_prepared/valid/inputs",
        target_dir="workspace/data_prepared/valid/targets",
        epochs=1,
        batch_size=8,
        image_size=(256,256),
        gen_base_features=64, gen_depth=2, gen_lr=1e-3,
        disc_base_features=64, disc_depth=2, disc_lr=1e-4,
        adv_weight=1.0, l1_weight=50.0,
        real_label_weight=0.9, fake_label_weight=0.1,
        checkpoint_dir="checkpoints",
        sample_count=4
    )
