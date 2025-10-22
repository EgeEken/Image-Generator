# train_pix2pix_v2.py
import os
import argparse
import random
import shutil
from pathlib import Path
import tempfile
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
        assert len(self.inputs) == len(self.targets), f"Inputs and targets length mismatch: {len(self.inputs)} vs {len(self.targets)}"
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
# Generator (U-Net) - parametric features
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=64, depth=4):
        super().__init__()
        # We'll build encoder/decoder dynamically based on depth
        self.depth = depth
        enc_layers = []
        in_ch = in_channels
        out_ch = features
        # first down layer (no batchnorm)
        enc_layers.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, 2, 1), nn.LeakyReLU(0.2)))
        prev = out_ch
        # subsequent down layers
        for i in range(1, depth):
            enc_layers.append(nn.Sequential(nn.Conv2d(prev, prev*2, 4, 2, 1),
                                            nn.BatchNorm2d(prev*2), nn.LeakyReLU(0.2)))
            prev = prev*2
        self.encoder = nn.ModuleList(enc_layers)

        # decoder (transpose conv). We'll mirror encoder.
        dec_layers = []
        for i in range(depth-1, 0, -1):
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(prev, prev//2, 4, 2, 1),
                                            nn.BatchNorm2d(prev//2), nn.ReLU()))
            prev = prev//2
        self.decoder = nn.ModuleList(dec_layers)
        # final combine conv to output channels
        self.final = nn.ConvTranspose2d(prev*2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc_outs = []
        out = x
        for layer in self.encoder:
            out = layer(out)
            enc_outs.append(out)
        # bottom is last enc_outs[-1]
        out = enc_outs[-1]
        # decode with skip connections
        for i, layer in enumerate(self.decoder):
            up = layer(out)
            skip = enc_outs[-2 - i]  # mirror
            out = torch.cat([up, skip], dim=1)
        out = self.final(out)
        return self.tanh(out)

# -----------------------------
# PatchGAN Discriminator (parametric)
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, features=64, depth=3):
        super().__init__()
        layers = []
        prev = in_channels
        cur = features
        # first block (no batchnorm)
        layers.append(nn.Sequential(nn.Conv2d(prev, cur, 4, 2, 1), nn.LeakyReLU(0.2)))
        prev = cur
        for i in range(1, depth):
            layers.append(nn.Sequential(nn.Conv2d(prev, prev*2, 4, 2, 1),
                                        nn.BatchNorm2d(prev*2), nn.LeakyReLU(0.2)))
            prev = prev*2
        # final conv to single channel patch map
        layers.append(nn.Conv2d(prev, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# -----------------------------
# Utilities
# -----------------------------
def save_image_tensor_as_uint8(tensor, path):
    """tensor is CxHxW with values in [0,1] or [-1,1]"""
    arr = tensor.detach().cpu().numpy()
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    # CxHxW -> HxWxC
    arr = np.transpose(arr, (1,2,0))
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

# -----------------------------
# Training
# -----------------------------
def train(
    input_dir, target_dir,
    epochs=10, batch_size=8, lr=2e-4, image_size=(256,256),
    gen_features=64, gen_depth=4,
    disc_features=64, disc_depth=3,
    samples_per_epoch=4,
    checkpoint_dir="checkpoints",
    resume=False,
    accumulation_steps=1,
    l1_weight=100.0,
    seed=42
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Datasets
    train_ds = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)
    # create a small validation split from the end (10% or 100 images min)
    val_count = max( max(1, len(train_ds)//10), min(100, len(train_ds)) )
    val_indices = list(range(len(train_ds)-val_count, len(train_ds)))
    train_indices = list(range(0, len(train_ds)-val_count))
    from torch.utils.data import Subset
    train_ds_sub = Subset(train_ds, train_indices)
    val_ds = Subset(train_ds, val_indices)

    train_loader = DataLoader(train_ds_sub, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Models
    G = UNetGenerator(in_channels=1, out_channels=3, features=gen_features, depth=gen_depth).to(device)
    D = PatchDiscriminator(in_channels=4, features=disc_features, depth=disc_depth).to(device)

    # Optionally resume
    start_epoch = 0
    if resume:
        # find latest checkpoint
        ckpts = sorted(Path(checkpoint_dir).glob("G_epoch*.pth"))
        if ckpts:
            latest = ckpts[-1]
            start_epoch = int(latest.stem.split("epoch")[-1])
            G.load_state_dict(torch.load(latest, map_location=device))
            D.load_state_dict(torch.load(str(latest).replace("G_epoch", "D_epoch"), map_location=device))
            print(f"Resumed from epoch {start_epoch}")

    # Optimizers (AdamW)
    opt_G = torch.optim.AdamW(G.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-2)
    opt_D = torch.optim.AdamW(D.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=1e-2)

    # Scheduler (optional cosine)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=max(1, epochs), eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=max(1, epochs), eta_min=1e-6)

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()

    # AMP
    scaler_G = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    # MLflow: set tracking URI via environment variable or default http://mlflow:5000
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sketch2img"))

    run = mlflow.start_run()
    # log hyperparameters
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "image_size": image_size,
        "gen_features": gen_features,
        "gen_depth": gen_depth,
        "disc_features": disc_features,
        "disc_depth": disc_depth,
        "samples_per_epoch": samples_per_epoch,
        "accumulation_steps": accumulation_steps,
        "l1_weight": l1_weight
    })

    # print interval: 10 prints per epoch
    print_interval = max(1, len(train_loader)//10)

    global_step = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        G.train(); D.train()
        running_G = 0.0
        running_D = 0.0
        for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # ---------------------
            # Train Discriminator
            # ---------------------
            D.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                real_in = torch.cat([x, y], dim=1)
                pred_real = D(real_in)
                loss_D_real = criterion_adv(pred_real, torch.ones_like(pred_real, device=device))

                fake = G(x).detach()
                fake_in = torch.cat([x, fake], dim=1)
                pred_fake = D(fake_in)
                loss_D_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake, device=device))

                loss_D = (loss_D_real + loss_D_fake) * 0.5

            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            opt_D.zero_grad()

            # ---------------------
            # Train Generator (with accumulation support)
            # ---------------------
            G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                fake = G(x)
                fake_in = torch.cat([x, fake], dim=1)
                pred_fake = D(fake_in)
                loss_G_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake, device=device))
                loss_G_l1 = criterion_l1(fake, y)
                loss_G = loss_G_adv + l1_weight * loss_G_l1
                loss_G = loss_G / accumulation_steps

            scaler_G.scale(loss_G).backward()
            if (i+1) % accumulation_steps == 0:
                scaler_G.step(opt_G)
                scaler_G.update()
                opt_G.zero_grad()

            running_G += loss_G.item() * accumulation_steps
            running_D += loss_D.item()

            global_step += 1
            if (i % print_interval) == 0:
                avg_G = running_G / (i+1)
                avg_D = running_D / (i+1)
                print(f"[Epoch {epoch+1}] Step {i}/{len(train_loader)} AvgLoss_G: {avg_G:.4f}, AvgLoss_D: {avg_D:.4f}")
                mlflow.log_metric("avg_loss_G", avg_G, step=global_step)
                mlflow.log_metric("avg_loss_D", avg_D, step=global_step)

        # end epoch: scheduler step
        scheduler_G.step()
        scheduler_D.step()

        # Validation pass (compute L1 on val set)
        G.eval()
        total_val_l1 = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device, non_blocking=True)
                vy = vy.to(device, non_blocking=True)
                pred = G(vx)
                total_val_l1 += criterion_l1(pred, vy).item()
                val_batches += 1
        val_loss = total_val_l1 / max(1, val_batches)
        print(f"Epoch {epoch+1} validation L1: {val_loss:.6f}")
        mlflow.log_metric("val_l1", val_loss, step=epoch)

        # Save checkpoint
        ckpt_G = os.path.join(checkpoint_dir, f"G_epoch{epoch+1}.pth")
        ckpt_D = os.path.join(checkpoint_dir, f"D_epoch{epoch+1}.pth")
        torch.save(G.state_dict(), ckpt_G)
        torch.save(D.state_dict(), ckpt_D)
        mlflow.log_artifact(ckpt_G, artifact_path=f"checkpoints/epoch_{epoch+1}")
        mlflow.log_artifact(ckpt_D, artifact_path=f"checkpoints/epoch_{epoch+1}")

        # Save sample outputs as epoch folders: epoch{n}/sample{i}/(input,output,expected)
        # We'll sample from val_loader
        tmp_root = Path(tempfile.mkdtemp(prefix=f"epoch_{epoch+1}_"))
        epoch_folder = tmp_root / f"epoch{epoch+1}"
        epoch_folder.mkdir(parents=True, exist_ok=True)
        sample_count = min(samples_per_epoch, len(val_ds))
        # choose evenly spaced indices from val set
        chosen = np.linspace(0, len(val_ds)-1, sample_count, dtype=int)
        for si, idx in enumerate(chosen):
            in_img, exp_img = val_ds[idx]
            in_img_batch = in_img.unsqueeze(0).to(device)
            with torch.no_grad():
                out_img = G(in_img_batch)[0]
            sample_dir = epoch_folder / f"sample{si+1}"
            sample_dir.mkdir()
            # save input (gray) as 3-channel for convenience
            in_save = (in_img.cpu().numpy() * 255).astype(np.uint8)
            if in_save.shape[0] == 1:
                in_save = np.transpose(in_save, (1,2,0))
                in_save = np.repeat(in_save, 3, axis=2)
            cv2.imwrite(str(sample_dir / "input.png"), cv2.cvtColor(in_save, cv2.COLOR_RGB2BGR))
            save_image_tensor_as_uint8(out_img, sample_dir / "output.png")
            save_image_tensor_as_uint8(exp_img, sample_dir / "expected_output.png")

        # log entire epoch folder to mlflow
        mlflow.log_artifacts(str(epoch_folder), artifact_path=f"epoch_{epoch+1}")
        # cleanup
        shutil.rmtree(tmp_root)

    mlflow.end_run()
    print("Training complete.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="/workspace/data_prepared/train/inputs")
    p.add_argument("--target_dir", type=str, default="/workspace/data_prepared/train/targets")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image_size", type=int, nargs=2, default=(256,256))
    p.add_argument("--gen_features", type=int, default=64)
    p.add_argument("--gen_depth", type=int, default=2)
    p.add_argument("--disc_features", type=int, default=64)
    p.add_argument("--disc_depth", type=int, default=2)
    p.add_argument("--samples_per_epoch", type=int, default=4)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--l1_weight", type=float, default=100.0)
    args = p.parse_args()

    train(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=tuple(args.image_size),
        gen_features=args.gen_features,
        gen_depth=args.gen_depth,
        disc_features=args.disc_features,
        disc_depth=args.disc_depth,
        samples_per_epoch=args.samples_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        accumulation_steps=args.accum,
        l1_weight=args.l1_weight
    )
