#!/usr/bin/env python3
"""
Stable Pix2Pix-style training script (GAN) tuned for reliable training.

Key improvements over original:
- Pretrain generator (L1/MSE) for a few epochs, then enable adversarial loss.
- Use Least-Squares GAN (LSGAN) for greater stability.
- Ramp up adversarial weight slowly across epochs.
- Weaken discriminator (dropout, label smoothing, noisy labels, update less often).
- Use GroupNorm for stability with batch=1.
- Reduced MLflow logging frequency to avoid GPU-CPU sync overhead.
- Many config params exposed via CLI.
"""
import os
import random
import shutil
from pathlib import Path
from PIL import Image
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils as tv_utils
import mlflow
import matplotlib.pyplot as plt

# -----------------------------
# Dataset
# -----------------------------
class Sketch2ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(256,256), exts=(".jpg",".jpeg",".png")):
        input_dir = Path(input_dir)
        target_dir = Path(target_dir)
        self.inputs = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
        self.targets = sorted([p for p in target_dir.rglob("*") if p.suffix.lower() in exts])
        if len(self.inputs) != len(self.targets):
            inputs_map = {p.name: p for p in self.inputs}
            targets_map = {p.name: p for p in self.targets}
            common = sorted(set(inputs_map.keys()) & set(targets_map.keys()))
            self.inputs = [inputs_map[k] for k in common]
            self.targets = [targets_map[k] for k in common]
        assert len(self.inputs) == len(self.targets) and len(self.inputs) > 0, \
            f"No matching input/target images found under {input_dir} and {target_dir} (found {len(self.inputs)} pairs)."
        self.image_size = image_size
        self.to_tensor_color = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
        self.to_tensor_gray  = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        in_p = str(self.inputs[idx])
        t_p  = str(self.targets[idx])
        input_img = Image.open(in_p).convert("L")
        target_img = Image.open(t_p).convert("RGB")
        input_img = self.to_tensor_gray(input_img)  # 1xH xW (0..1)
        target_img = self.to_tensor_color(target_img)  # 3xH xW (0..1)
        return input_img, target_img

# -----------------------------
# UNet generator (GroupNorm stable)
# -----------------------------
def _group_norm(channels):
    # choose groups: min(32, channels) but must divide channels. Choose divisor near 32.
    groups = min(32, channels)
    # reduce groups until divides channels
    while channels % groups != 0:
        groups -= 1
        if groups == 1:
            break
    return nn.GroupNorm(groups, channels)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=4, use_norm=True):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.depth = depth
        self.use_norm = use_norm
        self.encs = nn.ModuleList()
        self.encoder_channels = []

        prev_ch = in_channels
        for i in range(depth):
            out_ch = base_features * (2**i)
            layers = [nn.Conv2d(prev_ch, out_ch, 4, 2, 1)]
            # first encoder typically no norm, use LeakyReLU
            if use_norm and i != 0:
                layers.append(_group_norm(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.encs.append(nn.Sequential(*layers))
            self.encoder_channels.append(out_ch)
            prev_ch = out_ch

        # decoder
        self.decs = nn.ModuleList()
        cur_in = self.encoder_channels[-1]
        for enc_ch in reversed(self.encoder_channels[:-1]):
            out_ch = enc_ch
            layers = [nn.ConvTranspose2d(cur_in, out_ch, 4, 2, 1)]
            if use_norm:
                layers.append(_group_norm(out_ch))
            layers.append(nn.ReLU(inplace=True))
            self.decs.append(nn.Sequential(*layers))
            cur_in = out_ch * 2

        self.final = nn.ConvTranspose2d(cur_in, out_channels, 4, 2, 1)
        self.out_act = nn.Sigmoid()  # because we use 0..1 targets

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        for i, dec in enumerate(self.decs):
            out = dec(out)
            skip = skips[-(i+2)]
            if out.shape[2:] != skip.shape[2:]:
                out = nn.functional.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
        out = self.final(out)
        return self.out_act(out)

# -----------------------------
# Patch Discriminator (weakened + stable)
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_features=64, depth=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_channels
        for i in range(depth):
            out_ch = base_features * (2**i)
            layers.append(nn.Conv2d(prev, out_ch, 4, 2, 1))
            if i != 0:
                layers.append(_group_norm(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0 and i >= 1:
                layers.append(nn.Dropout2d(dropout))
            prev = out_ch
        # final conv to single-channel patch score map
        layers.append(nn.Conv2d(prev, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Helpers: LSGAN losses, label smoothing, noisy labels
# -----------------------------
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()

def real_label_tensor(shape, device, smooth=0.1, noise=0.0):
    # returns values in [1-smooth .. 1] plus small noise
    base = 1.0 - smooth
    t = torch.full(shape, base, device=device)
    if smooth > 0:
        t += (smooth * torch.rand_like(t))
    if noise > 0:
        t += (noise * (torch.rand_like(t) - 0.5))
    return t

def fake_label_tensor(shape, device, smooth=0.0, noise=0.0):
    # returns values near 0 with small noise
    t = torch.zeros(shape, device=device)
    if smooth > 0:
        t += (smooth * torch.rand_like(t))
    if noise > 0:
        t += (noise * torch.rand_like(t))
    return t

# -----------------------------
# Training function
# -----------------------------
def train(
    input_dir, target_dir,
    epochs=50, batch_size=4, image_size=(256,256),
    gen_base_features=64, gen_depth=4, gen_lr=2e-4, gen_wd=0,
    disc_base_features=64, disc_depth=3, disc_lr=2e-4, disc_wd=0,
    pretrain_epochs=3,
    adv_start_epoch=3, adv_ramp_epochs=20, adv_max=1.0,
    lambda_l1=100.0,
    d_updates_per_step=1, d_dropout=0.2,
    use_amp=True, device_pref=None,
    mlflow_uri=None, mlflow_log_every=20,
    preload_dataset=False, num_workers=2,
    checkpoint_dir="checkpoints", seed=42, resume=False
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = device_pref or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # MLflow
    mlflow_uri = mlflow_uri or os.environ.get("MLFLOW_TRACKING_URI", None)
    if mlflow_uri:
        mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sketch2img"))
    run = mlflow.start_run()
    run_id = run.info.run_id
    print("MLflow run id:", run_id)
    artifact_root = Path(mlflow.get_artifact_uri()).resolve()

    dataset = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)
    print(len(dataset), "pairs found")

    if preload_dataset:
        print("Preloading dataset into memory...")
        mem = [dataset[i] for i in range(len(dataset))]
        # replace dataset with simple in-memory wrapper
        class MemDS(torch.utils.data.Dataset):
            def __init__(self, arr): self.arr = arr
            def __len__(self): return len(self.arr)
            def __getitem__(self, i): return self.arr[i]
        dataset = MemDS(mem)

    if len(dataset) < 3:
        train_ds = dataset; val_ds = dataset
    else:
        val_count = max(1, len(dataset)//10)
        train_count = len(dataset) - val_count
        train_ds = Subset(dataset, list(range(train_count)))
        val_ds   = Subset(dataset, list(range(train_count, train_count + val_count)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0, num_workers-1), pin_memory=True)

    print(f"train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # Models
    G = UNetGenerator(in_channels=1, out_channels=3, base_features=gen_base_features, depth=gen_depth, use_norm=True).to(device)
    D = PatchDiscriminator(in_channels=4, base_features=disc_base_features, depth=disc_depth, dropout=d_dropout).to(device)

    print("G params:", sum(p.numel() for p in G.parameters()))
    print("D params:", sum(p.numel() for p in D.parameters()))

    opt_G = optim.Adam(G.parameters(), lr=gen_lr, betas=(0.5, 0.999), weight_decay=gen_wd)
    opt_D = optim.Adam(D.parameters(), lr=disc_lr, betas=(0.5, 0.999), weight_decay=disc_wd)

    sched_G = None
    sched_D = None

    scaler_G = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(use_amp and device.startswith("cuda")))

    # optionally resume (simple)
    start_epoch = 0
    if resume:
        # naive resume: try find latest checkpoint under checkpoint_dir
        ckpts = sorted(Path(checkpoint_dir).glob("G_epoch*.pth"))
        if ckpts:
            latest = ckpts[-1]
            start_epoch = int(latest.stem.split("epoch")[-1])
            print("Resuming from epoch", start_epoch)
            G.load_state_dict(torch.load(latest, map_location=device))
            Df = str(latest).replace("G_epoch", "D_epoch")
            if Path(Df).exists():
                D.load_state_dict(torch.load(Df, map_location=device))

    # log params
    mlflow.log_params({
        "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
        "gen_base_features": gen_base_features, "gen_depth": gen_depth, "gen_lr": gen_lr,
        "disc_base_features": disc_base_features, "disc_depth": disc_depth, "disc_lr": disc_lr,
        "pretrain_epochs": pretrain_epochs, "adv_start_epoch": adv_start_epoch, "adv_ramp_epochs": adv_ramp_epochs,
        "adv_max": adv_max, "lambda_l1": lambda_l1, "d_updates_per_step": d_updates_per_step,
        "mlflow_log_every": mlflow_log_every, "preload_dataset": preload_dataset
    })

    global_step = start_epoch * len(train_loader)

    def get_adv_weight(epoch):
        if epoch < adv_start_epoch:
            return 0.0
        # linearly ramp from 0 -> adv_max over adv_ramp_epochs
        t = min(1.0, (epoch - adv_start_epoch + 1) / max(1.0, adv_ramp_epochs))
        return float(t * adv_max)

    # training loop
    for epoch in range(start_epoch, epochs):
        G.train(); D.train()
        adv_w = get_adv_weight(epoch)
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        batch_count = 0
        start_time = time.time()

        for i, (xin, yin) in enumerate(train_loader):
            batch_count += 1
            xin = xin.to(device, non_blocking=True)
            yin = yin.to(device, non_blocking=True)

            # --------------------
            # Discriminator updates (perform d_updates_per_step times)
            # --------------------
            if adv_w > 0.0:
                for dstep in range(d_updates_per_step):
                    D.zero_grad()
                    with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                        # real
                        real_in = torch.cat([xin, yin], dim=1)
                        pred_real = D(real_in)
                        # fake
                        fake_imgs = G(xin).detach()
                        fake_in = torch.cat([xin, fake_imgs], dim=1)
                        pred_fake = D(fake_in)

                        # label smoothing + noise
                        real_t = real_label_tensor(pred_real.shape, device, smooth=0.1, noise=0.02)
                        fake_t = fake_label_tensor(pred_fake.shape, device, smooth=0.02, noise=0.02)

                        # LSGAN losses
                        loss_D_real = mse_loss(pred_real, real_t)
                        loss_D_fake = mse_loss(pred_fake, fake_t)
                        loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    scaler_D.scale(loss_D).backward()
                    scaler_D.step(opt_D)
                    scaler_D.update()
                    opt_D.zero_grad()
                    epoch_D_loss += loss_D.item()

            # --------------------
            # Generator update
            # --------------------
            G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(use_amp and device.startswith("cuda"))):
                fake = G(xin)
                # pixel loss (L1)
                loss_pix = l1_loss(fake, yin) * lambda_l1

                # adversarial loss only if adv_w>0
                if adv_w > 0.0:
                    fake_in = torch.cat([xin, fake], dim=1)
                    pred_fake_for_G = D(fake_in)
                    # generator wants D(fake) -> 1
                    real_label_for_G = real_label_tensor(pred_fake_for_G.shape, device, smooth=0.05, noise=0.01)
                    loss_adv = mse_loss(pred_fake_for_G, real_label_for_G)
                else:
                    loss_adv = torch.tensor(0.0, device=device)

                loss_G = loss_pix + adv_w * loss_adv

            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            opt_G.zero_grad()
            epoch_G_loss += loss_G.item()

            global_step += 1

            # mlflow logging throttle
            if (global_step % mlflow_log_every) == 0:
                # log lightweight metrics
                mlflow.log_metric("train_loss_G", epoch_G_loss / max(1, batch_count), step=global_step)
                if adv_w > 0:
                    mlflow.log_metric("train_loss_D", epoch_D_loss / max(1, batch_count), step=global_step)
                mlflow.log_metric("adv_weight", adv_w, step=global_step)

            # occasional console print
            if i % max(1, (len(train_loader)//10)) == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Step {i}/{len(train_loader)} | G_loss(avg): {epoch_G_loss/max(1,batch_count):.6f} | D_loss(avg): {epoch_D_loss/max(1,batch_count):.6f} | adv_w: {adv_w:.4f}")

        # end epoch: validation L1
        G.eval()
        val_l1 = 0.0
        val_batches = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                pred = G(vx)
                val_l1 += l1_loss(pred, vy).item()
                val_batches += 1
        val_l1 = val_l1 / max(1, val_batches)
        print(f"Epoch {epoch+1} finished in {time.time()-start_time:.1f}s. val L1: {val_l1:.6f}. adv_w: {adv_w:.4f}")

        # log epoch-level metrics
        mlflow.log_metric("val_l1", val_l1, step=epoch+1)
        mlflow.log_metric("epoch_G_loss", epoch_G_loss / max(1, batch_count), step=epoch+1)
        if adv_w > 0:
            mlflow.log_metric("epoch_D_loss", epoch_D_loss / max(1, batch_count), step=epoch+1)

        # Save checkpoints
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(G.state_dict(), ckpt_dir / f"G_epoch{epoch+1}.pth")
        torch.save(D.state_dict(), ckpt_dir / f"D_epoch{epoch+1}.pth")
        mlflow.log_artifact(str(ckpt_dir / f"G_epoch{epoch+1}.pth"), artifact_path=f"checkpoints/epoch_{epoch+1}")
        mlflow.log_artifact(str(ckpt_dir / f"D_epoch{epoch+1}.pth"), artifact_path=f"checkpoints/epoch_{epoch+1}")

        # Save small number of sample images (1 or few)
        samples_dir = Path(artifact_root) / f"samples/epoch_{epoch+1}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_count = min(4, len(dataset))
        chosen = np.linspace(0, len(dataset)-1, sample_count, dtype=int)
        for si, idx in enumerate(chosen):
            in_img, exp_img = dataset[idx]
            in_img_batch = in_img.unsqueeze(0).to(device)
            with torch.no_grad():
                out_img = G(in_img_batch)[0]
            sample_dir = samples_dir / f"sample{si+1}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            in_save = (in_img.cpu().numpy() * 255).astype(np.uint8)
            if in_save.shape[0] == 1:
                in_save = np.transpose(in_save, (1,2,0))
                in_save = np.repeat(in_save, 3, axis=2)
            from PIL import Image as PILImage
            PILImage.fromarray(in_save[:,:,0]).convert("RGB").save(sample_dir / "input.png")
            out_arr = (out_img.cpu().numpy() * 255).astype(np.uint8)
            out_arr = np.transpose(out_arr, (1,2,0))
            PILImage.fromarray(out_arr).save(sample_dir / "output.png")
            exp_arr = (exp_img.cpu().numpy() * 255).astype(np.uint8)
            exp_arr = np.transpose(exp_arr, (1,2,0))
            PILImage.fromarray(exp_arr).save(sample_dir / "expected.png")

        mlflow.log_artifacts(str(samples_dir), artifact_path=f"samples/epoch_{epoch+1}")

    mlflow.end_run()
    print("Training complete.")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(base/"data_prepared"/"valid"/"inputs"))
    parser.add_argument("--target_dir", type=str, default=str(base/"data_prepared"/"valid"/"targets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=(128,128))
    parser.add_argument("--gen_base_features", type=int, default=64)
    parser.add_argument("--gen_depth", type=int, default=4)
    parser.add_argument("--gen_lr", type=float, default=2e-4)
    parser.add_argument("--gen_wd", type=float, default=0)
    parser.add_argument("--disc_base_features", type=int, default=64)
    parser.add_argument("--disc_depth", type=int, default=3)
    parser.add_argument("--disc_lr", type=float, default=2e-4)
    parser.add_argument("--disc_wd", type=float, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=5)
    parser.add_argument("--adv_start_epoch", type=int, default=5)
    parser.add_argument("--adv_ramp_epochs", type=int, default=20)
    parser.add_argument("--adv_max", type=float, default=0.9)
    parser.add_argument("--lambda_l1", type=float, default=100.0)
    parser.add_argument("--d_updates_per_step", type=int, default=1)
    parser.add_argument("--d_dropout", type=float, default=0.2)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--mlflow_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", None))
    parser.add_argument("--mlflow_log_every", type=int, default=20)
    parser.add_argument("--preload_dataset", action="store_true")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=tuple(args.image_size),
        gen_base_features=args.gen_base_features,
        gen_depth=args.gen_depth,
        gen_lr=args.gen_lr,
        gen_wd=args.gen_wd,
        disc_base_features=args.disc_base_features,
        disc_depth=args.disc_depth,
        disc_lr=args.disc_lr,
        disc_wd=args.disc_wd,
        pretrain_epochs=args.pretrain_epochs,
        adv_start_epoch=args.adv_start_epoch,
        adv_ramp_epochs=args.adv_ramp_epochs,
        adv_max=args.adv_max,
        lambda_l1=args.lambda_l1,
        d_updates_per_step=args.d_updates_per_step,
        d_dropout=args.d_dropout,
        use_amp=args.use_amp,
        mlflow_uri=args.mlflow_uri,
        mlflow_log_every=args.mlflow_log_every,
        preload_dataset=args.preload_dataset,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume
    )
