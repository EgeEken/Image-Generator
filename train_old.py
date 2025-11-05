import os
import argparse
import random
import shutil
from pathlib import Path
import cv2
from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import mlflow

# -----------------------------
# Dataset (recursive)
# -----------------------------
class Sketch2ImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, image_size=(256,256), exts=(".jpg",".jpeg",".png")):
        input_dir = Path(input_dir)
        target_dir = Path(target_dir)
        # recursively gather images
        self.inputs = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in exts])
        self.targets = sorted([p for p in target_dir.rglob("*") if p.suffix.lower() in exts])
        if len(self.inputs) != len(self.targets):
            # try to match 1:1 by filename if folder structure differs (best-effort)
            inputs_map = {p.name: p for p in self.inputs}
            targets_map = {p.name: p for p in self.targets}
            common = sorted(set(inputs_map.keys()) & set(targets_map.keys()))
            self.inputs = [inputs_map[k] for k in common]
            self.targets = [targets_map[k] for k in common]
        assert len(self.inputs) == len(self.targets) and len(self.inputs) > 0, \
            f"No matching input/target images found under {input_dir} and {target_dir} (found {len(self.inputs)} pairs)."
        self.image_size = image_size

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        in_p = str(self.inputs[idx])
        t_p  = str(self.targets[idx])
        input_img = cv2.imread(in_p, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(t_p, cv2.IMREAD_COLOR)

        input_img = cv2.resize(input_img, self.image_size, interpolation=cv2.INTER_AREA)
        target_img = cv2.resize(target_img, self.image_size, interpolation=cv2.INTER_AREA)

        input_img = torch.tensor(input_img/255.0, dtype=torch.float32).unsqueeze(0)  # 1xHxW
        target_img = torch.tensor(target_img/255.0, dtype=torch.float32).permute(2,0,1)  # 3xHxW

        return input_img, target_img

# -----------------------------
# U-Net generator (dynamic, correct channel bookkeeping)
# -----------------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, base_features=64, depth=4):
        super().__init__()
        assert depth >= 1, "depth must be >=1"
        self.depth = depth
        self.encs = nn.ModuleList()
        self.encoder_channels = []

        prev_ch = in_channels
        # encoder: double channels each step
        for i in range(depth):
            out_ch = base_features * (2**i)
            if i == 0:
                block = nn.Sequential(nn.Conv2d(prev_ch, out_ch, 4, 2, 1), nn.ReLU(0.2))
            else:
                block = nn.Sequential(nn.Conv2d(prev_ch, out_ch, 4, 2, 1),
                                      nn.InstanceNorm2d(out_ch), nn.ReLU(0.2))
            self.encs.append(block)
            self.encoder_channels.append(out_ch)
            prev_ch = out_ch

        # decoder: we construct layers so that each ConvTranspose2d maps:
        # current_in -> next_out where next_out equals the encoder channel of the layer we will concat with.
        self.decs = nn.ModuleList()
        cur_in = self.encoder_channels[-1]  # bottom channels
        # iterate backwards over encoder channels (skip the bottom one), for each create ConvTranspose2d(cur_in, out_ch)
        for enc_ch in reversed(self.encoder_channels[:-1]):
            out_ch = enc_ch
            block = nn.Sequential(
                nn.ConvTranspose2d(cur_in, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU()
            )
            self.decs.append(block)
            # after concat, next cur_in becomes out_ch * 2
            cur_in = out_ch * 2

        # final layer: after last concat, cur_in is base_features*2 (since we concat with first encoder output)
        self.final = nn.ConvTranspose2d(cur_in, out_channels, 4, 2, 1)
        self.out_act = nn.Sigmoid()
        # self.out_act = nn.Tanh()

    def forward(self, x):
        skips = []
        out = x
        for enc in self.encs:
            out = enc(out)
            skips.append(out)
        # decode
        for i, dec in enumerate(self.decs):
            out = dec(out)
            skip = skips[-(i+2)]  # match encoder
            if out.shape[2:] != skip.shape[2:]:
                out = nn.functional.interpolate(out, size=skip.shape[2:], mode="bilinear", align_corners=False)
            out = torch.cat([out, skip], dim=1)
        out = self.final(out)
        return self.out_act(out)

# -----------------------------
# PatchGAN discriminator (parametric)
# -----------------------------
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=4, base_features=64, depth=3):
        super().__init__()
        assert depth >= 1
        layers = []
        prev = in_channels
        for i in range(depth):
            out_ch = base_features * (2**i)
            if i == 0:
                layers.append(nn.Conv2d(prev, out_ch, 4, 2, 1))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Conv2d(prev, out_ch, 4, 2, 1))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU(0.2))
            prev = out_ch
        layers.append(nn.Conv2d(prev, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------------
# helper to save images
# -----------------------------
def save_tensor_image(tensor, path):
    # tensor: CxHxW in [-1,1] or [0,1]
    arr = tensor.detach().cpu().numpy()
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (1,2,0))
    cv2.imwrite(str(path), cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

# -----------------------------
# Training function (parametrized)
# -----------------------------
def train(
    input_dir, target_dir,
    epochs=10, batch_size=4, image_size=(256,256),
    gen_base_features=64, gen_depth=4, gen_lr=2e-4, gen_wd=1e-2,
    disc_base_features=64, disc_depth=3, disc_lr=1e-4, disc_wd=1e-2,
    adv_weight=1.0, l1_weight=50.0,
    real_label=0.9, fake_label=0.1, d_updates_per_step=1,
    prints_per_epoch=10, samples_per_epoch=4,
    accumulation_steps=1, num_workers=2,
    checkpoint_dir="checkpoints", mlflow_uri=None, seed=42, resume=False
):
    random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    base_dir = Path(__file__).resolve().parent
    # default mlflow location if not provided
    if mlflow_uri is None:
        mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", f"file:{base_dir/'mlruns'}")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "sketch2img"))

    # dataset (recursive)
    dataset = Sketch2ImageDataset(input_dir, target_dir, image_size=image_size)

    print(len(dataset), "image pairs found.")

    if len(dataset) < 3:
        # dont split into validation and training, make val and train the same
        train_ds = dataset
        val_ds = dataset
    else:
        # split small validation (10% or min 1)
        val_count = max(1, len(dataset)//10)
        train_count = len(dataset) - val_count
        indices = list(range(len(dataset)))
        train_idx = indices[:train_count]
        val_idx = indices[train_count:]
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=max(0,num_workers-1), pin_memory=True)

    print(f"Dataset loaded - train samples: {len(train_ds)}, val samples: {len(val_ds)}")

    # Models
    G = UNetGenerator(in_channels=1, out_channels=3, base_features=gen_base_features, depth=gen_depth).to(device)
    D = PatchDiscriminator(in_channels=4, base_features=disc_base_features, depth=disc_depth).to(device)

    print(f"Models loaded - G params: {sum(p.numel() for p in G.parameters())}, D params: {sum(p.numel() for p in D.parameters())}")

    # optional resume (load latest checkpoint if resume True)
    start_epoch = 0
    if resume:
        ckpts = sorted(Path(checkpoint_dir).glob("G_epoch*.pth"))
        if ckpts:
            latest = ckpts[-1]
            start_epoch = int(latest.stem.split("epoch")[-1])
            print(f"Resuming from epoch {start_epoch}")
            G.load_state_dict(torch.load(latest, map_location=device))
            D.load_state_dict(torch.load(str(latest).replace("G_epoch", "D_epoch"), map_location=device))

    # optimizers
    opt_G = torch.optim.AdamW(G.parameters(), lr=gen_lr, betas=(0.9,0.95), weight_decay=gen_wd)
    opt_D = torch.optim.AdamW(D.parameters(), lr=disc_lr, betas=(0.9,0.95), weight_decay=disc_wd)
    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=max(1,epochs), eta_min=1e-6)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=max(1,epochs), eta_min=1e-6)

    # losses
    l1_loss = nn.L1Loss()
    adv_loss = nn.BCEWithLogitsLoss()

    # AMP
    scaler_G = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))
    scaler_D = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

    # MLflow run
    run = mlflow.start_run()
    run_id = run.info.run_id
    print(f"MLflow run ID: {run_id}")
    artifact_root = Path(mlflow.get_artifact_uri()).resolve()
    mlflow.log_params({
        "dataset_path": f"{input_dir}",
        "epochs": epochs, "batch_size": batch_size, "image_size": image_size,
        "gen_base_features": gen_base_features, "gen_depth": gen_depth, "gen_lr": gen_lr, "gen_wd": gen_wd,
        "disc_base_features": disc_base_features, "disc_depth": disc_depth, "disc_lr": disc_lr, "disc_wd": disc_wd,
        "adv_weight": adv_weight, "l1_weight": l1_weight,
        "real_label": real_label, "fake_label": fake_label, "d_updates_per_step": d_updates_per_step,
        "prints_per_epoch": prints_per_epoch, "samples_per_epoch": samples_per_epoch
    })

    print_interval = max(1, len(train_loader)//prints_per_epoch)
    total_steps = start_epoch * len(train_loader)
    for epoch in range(start_epoch, epochs):
        G.train(); D.train()
        running_G = 0.0; running_D = 0.0
        for i, (xin, yin) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            xin = xin.to(device, non_blocking=True)
            yin = yin.to(device, non_blocking=True)

            # --- Discriminator updates (can run multiple steps per generator step if desired) ---
            for dstep in range(d_updates_per_step):
                D.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                    real_in = torch.cat([xin, yin], dim=1)
                    pred_real = D(real_in)
                    # smooth labels
                    real_targets = torch.full_like(pred_real, real_label, device=device)
                    loss_D_real = adv_loss(pred_real, real_targets)

                    fake = G(xin).detach()
                    fake_in = torch.cat([xin, fake], dim=1)
                    pred_fake = D(fake_in)
                    fake_targets = torch.full_like(pred_fake, fake_label, device=device)
                    loss_D_fake = adv_loss(pred_fake, fake_targets)

                    loss_D = 0.5*(loss_D_real + loss_D_fake)

                scaler_D.scale(loss_D).backward()
                scaler_D.step(opt_D)
                scaler_D.update()
                opt_D.zero_grad()

            # --- Generator update ---
            G.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device=="cuda")):
                fake = G(xin)
                fake_in = torch.cat([xin, fake], dim=1)
                pred_fake_for_G = D(fake_in)
                g_adv = adv_loss(pred_fake_for_G, torch.full_like(pred_fake_for_G, real_label, device=device))
                g_l1  = l1_loss(fake, yin)
                g_total = adv_weight * g_adv + l1_weight * g_l1
                g_total = g_total / (adv_weight + l1_weight)
            scaler_G.scale(g_total).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            opt_G.zero_grad()

            running_G += g_total.item()
            running_D += loss_D.item()
            total_steps += 1

            mlflow.log_metric("train_loss_G", running_G/(i+1), step=total_steps)
            mlflow.log_metric("train_loss_D", running_D/(i+1), step=total_steps)

            # prints only prints_per_epoch times
            if i % print_interval == 0:
                avg_G = running_G / (i+1)
                avg_D = running_D / (i+1)
                print(f"[Epoch {epoch+1}] Step {i}/{len(train_loader)} AvgLoss_G: {avg_G:.4f}, AvgLoss_D: {avg_D:.4f}")

        # end epoch: validation pass
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
        print(f"Epoch {epoch+1} val L1: {val_l1:.6f}")
        mlflow.log_metric("val_l1", val_l1, step=epoch+1)

        # Save checkpoints under the MLflow run's artifact directory
        run_ckpt_dir = artifact_root / f"checkpoints/epoch_{epoch+1}"
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_G = run_ckpt_dir / f"G_epoch{epoch+1}.pth"
        ckpt_D = run_ckpt_dir / f"D_epoch{epoch+1}.pth"
        torch.save(G.state_dict(), ckpt_G)
        torch.save(D.state_dict(), ckpt_D)
        mlflow.log_artifacts(str(run_ckpt_dir), artifact_path=f"checkpoints/epoch_{epoch+1}")

        # Save limited number of samples (no mid-epoch spam)
        samples_dir = artifact_root / f"samples/epoch_{epoch+1}"
        samples_dir.mkdir(parents=True, exist_ok=True)
        sample_count = min(samples_per_epoch, len(dataset))
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
            cv2.imwrite(str(sample_dir / "input.png"), cv2.cvtColor(in_save, cv2.COLOR_RGB2BGR))
            save_tensor_image(out_img, sample_dir / "output.png")
            save_tensor_image(exp_img, sample_dir / "expected_output.png")

        mlflow.log_artifacts(str(samples_dir), artifact_path=f"samples/epoch_{epoch+1}")
        # log the first sample output.png (samples/epoch_{epoch+1}/sample1/output.png) as main sample of the epoch, save it as output_epoch{epoch+1}.png
        # cant use artifact_path to name it, that would create a folder
        # log it as a figure using mlflow.log_figure and matplotlib
        first_sample_output = samples_dir / "sample1" / "output.png"
        fig, ax = plt.subplots(figsize=(6,6))
        img = cv2.imread(str(first_sample_output))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        mlflow.log_figure(fig, f"output_epoch{epoch+1}.png")
        plt.close(fig)
        shutil.rmtree(artifact_root)

        # schedulers
        sched_G.step(); sched_D.step()

    mlflow.end_run()
    print("Training finished.")

# -----------------------------
# CLI / Run
# -----------------------------
if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(base/"data_prepared"/"ultra_overfit"/"inputs"))
    parser.add_argument("--target_dir", type=str, default=str(base/"data_prepared"/"ultra_overfit"/"targets"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, nargs=2, default=(128,128))
    parser.add_argument("--gen_base_features", type=int, default=64)
    parser.add_argument("--gen_depth", type=int, default=2)
    parser.add_argument("--gen_lr", type=float, default=2e-4)
    parser.add_argument("--gen_wd", type=float, default=1e-4)
    parser.add_argument("--disc_base_features", type=int, default=32)
    parser.add_argument("--disc_depth", type=int, default=2)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--disc_wd", type=float, default=1e-2)
    parser.add_argument("--adv_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=50.0)
    parser.add_argument("--real_label", type=float, default=0.8)
    parser.add_argument("--fake_label", type=float, default=0.2)
    parser.add_argument("--d_updates_per_step", type=int, default=1)
    parser.add_argument("--prints_per_epoch", type=int, default=1)
    parser.add_argument("--samples_per_epoch", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default=str(base/"checkpoints"))
    parser.add_argument("--mlflow_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI", f"file:{base/'mlruns'}"))
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
        adv_weight=args.adv_weight,
        l1_weight=args.l1_weight,
        real_label=args.real_label,
        fake_label=args.fake_label,
        d_updates_per_step=args.d_updates_per_step,
        prints_per_epoch=args.prints_per_epoch,
        samples_per_epoch=args.samples_per_epoch,
        checkpoint_dir=args.checkpoint_dir,
        mlflow_uri=args.mlflow_uri,
        resume=args.resume
    )
