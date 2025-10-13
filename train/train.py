# train/train.py
import os
import argparse
import time
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    return p.parse_args()

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(10, 1)
    def forward(self, x):
        return self.l(x)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "default")
    mlflow.set_experiment(experiment_name)

    # Example parameters to log
    with mlflow.start_run():
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("lr", args.lr)

        # ---- Replace this block with your pix2pix-turbo training loop ----
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        loss_fn = nn.MSELoss()
        for epoch in range(args.epochs):
            # dummy training loop (replace with real dataloader & train)
            input = torch.randn(args.batch_size, 10)
            target = torch.randn(args.batch_size, 1)
            pred = model(input)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mlflow.log_metric("train_loss", loss.item(), step=epoch)
            print(f"[epoch {epoch}] loss={loss.item():.4f}")

        # Save model locally
        model_path = os.path.join(args.output_dir, "model.pt")
        torch.save(model.state_dict(), model_path)

        # Log artifacts and model with MLflow
        mlflow.log_artifact(model_path, artifact_path="models")
        # Optionally use mlflow.pytorch log_model to preserve format
        mlflow.pytorch.log_model(model, artifact_path="pytorch_model")
        # Save a small text file with info
        info_path = os.path.join(args.output_dir, "info.txt")
        with open(info_path, "w") as f:
            f.write(f"Trained for {args.epochs} epochs\n")
        mlflow.log_artifact(info_path)

    print("Training complete. MLflow URI:", mlflow_uri)

if __name__ == "__main__":
    main()
