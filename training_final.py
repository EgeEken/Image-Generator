import os
import random
import shutil
from pathlib import Path
import tempfile
import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
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
# Generator (U-Net)
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=2):
        super().__init__()
        self.depth = depth
        enc_layers = []
        in_ch = in_channels
        out_ch = base_features
        enc_layers.append(nn.Sequential(nn.Conv2d(in_ch, out_ch, 4, 2, 1), nn.LeakyReLU(0.2)))
        prev = out_ch
        for i in range(1, depth):
            enc_layers.append(nn.Sequential(nn.Conv2d(prev, prev * 2, 4, 2, 1),
                                            nn.BatchNorm2d(prev * 2), nn.LeakyReLU(0.2)))
            prev = prev * 2
        self.encoder = nn.ModuleList(enc_layers)

        dec_layers = []
        for i in range(depth - 1, 0, -1):
            dec_layers.append(nn.Sequential(nn.ConvTranspose2d(prev, prev // 2, 4, 2, 1),
                                            nn.BatchNorm2d(prev // 2), nn.ReLU()))
            prev = prev // 2
        self.decoder = nn.ModuleList(dec_layers)
        self.final = nn.ConvTranspose2d(prev * 2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        enc_outs = []
        out = x
        for layer in self.encoder:
            out = layer(out)
            enc_outs.append(out)
        out = enc_outs[-1]
        for i, layer in enumerate(self.decoder):
            up = layer(out)
            skip = enc_outs[-2 - i]
            out = torch.cat([up, skip], dim=1)
        out = self.final(out)
        return self.tanh(out)

# -----------------------------
# Discriminator (PatchGAN)
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_features=64, depth=2):
        super().__init__()
        layers = []
        prev = in_channels
        cur = base_features
        layers.append(nn.Sequential(nn.Conv2d(prev, cur, 4, 2, 1), nn.LeakyReLU(0.2)))
        prev = cur
        for i in range(1, depth):
            layers.append(nn.Sequential(nn.Conv2d(prev, prev * 2, 4, 2, 1),
                                        nn.BatchNorm2d(prev * 2), nn.LeakyReLU(0.2)))
            prev = prev * 2
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
    epochs=1, batch_size=8, image_size=(256, 256),
    gen_features=64, gen_depth=2, gen_lr=2e-4, gen_weight_decay=1e-2,
    disc_features=64, disc_depth=2, disc_lr=2e-4, disc_weight_decay=1e-2,
    adv_weight=1.0, l1_weight=100.0,
    real_label_weight=1.0, fake_label_weight=0.0,
    checkpoint_dir="checkpoints",
    samples_per_epoch=4,
    prints_per_epoch=10,
    seed=42
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # check if dataset is found on input_dir and target_dir
    if not os.path.exists(input_dir) or not os.path.exists(target_dir):
        raise FileNotFoundError(f"Input or target directory not found: {input_dir}, {target_dir}")
    

    # Datasets
    dataset = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Models
    G = UNetGenerator(in_channels=1, out_channels=3, base_features=gen_features, depth=gen_depth).to(device)
    D = PatchDiscriminator(in_channels=4, base_features=disc_features, depth=disc_depth).to(device)

    # Optionally resume
    start_epoch = 0

    # Optimizers (AdamW)
    opt_G = torch.optim.AdamW(G.parameters(), lr=gen_lr, betas=(0.9, 0.95), weight_decay=gen_weight_decay)
    opt_D = torch.optim.AdamW(D.parameters(), lr=disc_lr, betas=(0.9, 0.95), weight_decay=disc_weight_decay)

    # Scheduler (optional cosine)
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=max(1, epochs), eta_min=1e-6)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=max(1, epochs), eta_min=1e-6)

    # Losses
    criterion_l1 = nn.L1Loss()
    criterion_adv = nn.BCEWithLogitsLoss()

    # AMP
    scaler_G = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    run = mlflow.start_run()
    # log hyperparameters
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": image_size,
        "gen_features": gen_features,
        "gen_depth": gen_depth,
        "gen_lr": gen_lr,
        "gen_weight_decay": gen_weight_decay,
        "disc_features": disc_features,
        "disc_depth": disc_depth,
        "disc_lr": disc_lr,
        "disc_weight_decay": disc_weight_decay,
        "samples_per_epoch": samples_per_epoch,
        "adv_weight": adv_weight,
        "l1_weight": l1_weight,
        "real_label_weight": real_label_weight,
        "fake_label_weight": fake_label_weight,
    })

    # print interval: prints per epoch
    print_interval = max(1, len(loader)// prints_per_epoch)

    global_step = start_epoch * len(loader)
    for epoch in range(start_epoch, epochs):
        G.train(); D.train()
        running_G = 0.0
        running_D = 0.0
        for i, (x, y) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            D.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                real_in = torch.cat([x, y], dim=1)
                pred_real = D(real_in)
                loss_D_real = criterion_adv(pred_real, torch.full_like(pred_real, real_label_weight, device=device))

                fake = G(x).detach()
                fake_in = torch.cat([x, fake], dim=1)
                pred_fake = D(fake_in)
                loss_D_fake = criterion_adv(pred_fake, torch.full_like(pred_fake, fake_label_weight, device=device))

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
                loss_G_adv = criterion_adv(pred_fake, torch.full_like(pred_fake, real_label_weight, device=device))
                loss_G_l1 = criterion_l1(fake, y)
                loss_G = adv_weight * loss_G_adv + l1_weight * loss_G_l1
                loss_G = loss_G

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            opt_G.zero_grad()

            running_G += loss_G.item()
            running_D += loss_D.item()

            global_step += 1
            if (i % print_interval) == 0:
                avg_G = running_G / (i+1)
                avg_D = running_D / (i+1)
                print(f"[Epoch {epoch+1}] Step {i}/{len(loader)} AvgLoss_G: {avg_G:.4f}, AvgLoss_D: {avg_D:.4f}")
                mlflow.log_metric("avg_loss_G", avg_G, step=global_step)
                mlflow.log_metric("avg_loss_D", avg_D, step=global_step)

        # end epoch: scheduler step
        scheduler_G.step()
        scheduler_D.step()

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
        sample_count = min(samples_per_epoch, len(dataset))
        # choose evenly spaced indices from train set
        chosen = np.linspace(0, len(dataset)-1, sample_count, dtype=int)
        for si, idx in enumerate(chosen):
            in_img, exp_img = dataset[idx]
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
# Main
# -----------------------------
base_dir = Path(__file__).resolve().parent
input_dir = base_dir / "data_prepared/overfit/inputs"
target_dir = base_dir / "data_prepared/overfit/targets"

if __name__ == "__main__":
    train(
        input_dir=input_dir,
        target_dir=target_dir,
        epochs=10,
        batch_size=4,
        image_size=(256,256),
        gen_features=256, gen_depth=4, gen_lr=1e-3,
        disc_features=256, disc_depth=4, disc_lr=1e-4,
        adv_weight=1.0, l1_weight=50.0,
        real_label_weight=0.9, fake_label_weight=0.1,
        checkpoint_dir="checkpoints",
        samples_per_epoch=1,
        prints_per_epoch=1,
        seed=1234
    )
