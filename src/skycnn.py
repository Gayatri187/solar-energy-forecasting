"""The sky-image CNN, its dataset, and the training loop."""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.skyimages import NUMERIC_FEATURES, smart_persistence

SCALE = 10.0   # corrections are divided by this, so the network works with numbers around -1..1


def pick_device() -> torch.device:
    """Use the Apple-chip GPU (MPS) on a Mac, an NVIDIA GPU if present, otherwise the CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SkyDataset(torch.utils.data.Dataset):
    """Serves (6-channel image stack, numeric features, correction target) for each sample."""

    def __init__(self, samples: pd.DataFrame, images: np.ndarray, position: dict, mean: pd.Series, std: pd.Series):
        self.now = np.array([position[t] for t in samples["img_now"]])
        self.before = np.array([position[t] for t in samples["img_before"]])
        self.images = images
        self.x = ((samples[NUMERIC_FEATURES] - mean) / std).to_numpy(np.float32)
        self.sp = smart_persistence(samples).astype(np.float32)
        # Residual learning: the network predicts how much to CORRECT smart persistence
        self.y = ((samples["target"].to_numpy() - self.sp) / SCALE).astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        stack = np.concatenate([self.images[self.now[i]], self.images[self.before[i]]], axis=2)
        img = torch.from_numpy(stack).permute(2, 0, 1).float() / 255.0     # 6 x 64 x 64, values 0..1
        return img, torch.from_numpy(self.x[i]), torch.tensor(self.y[i])


class SkyCNN(nn.Module):
    """Two branches: a CNN reads the sky images, a small MLP reads the numbers. Then both are combined."""

    def __init__(self, n_numeric: int):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2))

        self.image_branch = nn.Sequential(
            block(6, 24), block(24, 48), block(48, 64),          # 64x64 -> 32 -> 16 -> 8
            nn.AdaptiveAvgPool2d(4), nn.Flatten(),                # 64 x 4 x 4 = 1024 numbers
            nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.3))
        self.numeric_branch = nn.Sequential(nn.Linear(n_numeric, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, img, x):
        return self.head(torch.cat([self.image_branch(img), self.numeric_branch(x)], dim=1)).squeeze(1)


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    """Final forecast in kW = smart persistence + predicted correction (never below 0)."""
    model.eval()
    out = [model(img.to(device), x.to(device)).cpu().numpy() for img, x, _ in loader]
    return np.clip(np.concatenate(out) * SCALE + loader.dataset.sp, 0, None)


def train(model, train_loader, val_loader, val_target, device, max_epochs=12, patience=3, lr=3e-4):
    """Train with early stopping: keep the weights with the best validation RMSE."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        start, total, n = time.time(), 0.0, 0
        for img, x, y in train_loader:
            img, x, y = img.to(device), x.to(device), y.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(img, x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        val_rmse = float(np.sqrt(np.mean((predict(model, val_loader, device) - val_target) ** 2)))
        history.append({"epoch": epoch, "train_loss": total / n, "val_rmse": val_rmse})
        print(f"epoch {epoch:2d}  train loss {total / n:.4f}  val RMSE {val_rmse:.3f} kW  ({time.time() - start:.0f}s)")
        if val_rmse < best:
            best, best_state, bad = val_rmse, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("stopping early: no improvement for", patience, "epochs")
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)
