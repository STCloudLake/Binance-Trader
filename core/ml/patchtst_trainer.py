"""PatchTST trainer — sequence preparation + training loop."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from core.ml.patchtst_model import PatchTSTModel, cross_entropy_loss
from core.ml.features import compute_features, REQUIRED_INDICATORS


class PatchTSTTrainer:
    """PatchTST trainer with expanding-window normalization."""

    def __init__(self,
                 data_dir: str,
                 seq_len: int = 100,
                 patch_len: int = 16,
                 stride: int = 8,
                 d_model: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.15,
                 device: str | None = None):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        if device is None:
            if torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda(); self.device = "cuda"
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

    # ── Data Preparation ─────────────────────────────────────────────

    def prepare_sequences(self, df: pd.DataFrame,
                          feature_cols: list[str] | None = None,
                          label_col: str = "label") -> tuple[torch.Tensor, torch.Tensor]:
        if feature_cols is None:
            feature_cols = [c for c in df.columns
                          if c not in (label_col,) and df[c].dtype in ('float64', 'float32', 'int64')]

        data = df[feature_cols].values.astype(np.float32)
        labels = df[label_col].values.astype(np.float32)

        valid_mask = ~(np.isnan(data).any(axis=1) | np.isnan(labels))
        data = data[valid_mask]
        labels = labels[valid_mask]

        if len(data) <= self.seq_len:
            return (torch.empty(0, self.seq_len, len(feature_cols)),
                    torch.empty(0, dtype=torch.long))

        num_samples = len(data) - self.seq_len
        # Expanding-window normalization: for each window i, use mean/std
        # of all data [0 : i+seq_len]. Preserves trend/regime information.
        X_list = []
        for i in range(num_samples):
            window = data[i:i + self.seq_len].copy()
            full_up_to = data[:i + self.seq_len]
            f_mean = full_up_to.mean(axis=0)
            f_std = full_up_to.std(axis=0) + 1e-8
            X_list.append((window - f_mean) / f_std)
        X = np.stack(X_list, axis=0)
        y = labels[self.seq_len:]

        return torch.tensor(X), torch.tensor(y, dtype=torch.long)

    # ── Training ─────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame,
              feature_cols: list[str] | None = None,
              label_col: str = "label",
              epochs: int = 60,
              batch_size: int = 64,
              learning_rate: float = 1e-3,
              validation_split: float = 0.2,
              patience: int = 15) -> tuple[Optional[PatchTSTModel], dict]:
        X, y = self.prepare_sequences(df, feature_cols, label_col)
        if len(X) < 60:
            return None, {"error": f"Insufficient sequences: {len(X)}"}

        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        n_classes = 3  # up, down, timeout
        num_features = X.shape[2]
        model = PatchTSTModel(
            num_features=num_features, seq_len=self.seq_len,
            patch_len=self.patch_len, stride=self.stride,
            d_model=self.d_model, num_heads=self.num_heads,
            num_layers=self.num_layers, dropout=self.dropout,
            output_mode="triple_barrier",
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_state = None
        patience_counter = 0

        train_ds = torch.utils.data.TensorDataset(X_train, y_train)
        val_ds = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = model(batch_X)
                loss = cross_entropy_loss(out, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            train_loss /= max(len(train_loader), 1)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    out = model(batch_X)
                    loss = cross_entropy_loss(out, batch_y)
                    val_loss += loss.item()
                    preds = out["probs"].argmax(dim=-1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += len(batch_y)
            val_loss /= max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.debug(f"PatchTST epoch {epoch+1}/{epochs}: "
                           f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            if patience_counter >= patience:
                logger.debug(f"PatchTST early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = {
            "val_loss": best_val_loss,
            "val_accuracy": best_val_acc,
            "epochs_trained": epoch + 1,
            "num_sequences": len(X),
            "num_features": num_features,
            "device": self.device,
        }
        return model, metrics

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, model: PatchTSTModel,
                df: pd.DataFrame,
                feature_cols: list[str] | None = None) -> dict | None:
        if len(df) < self.seq_len:
            return None

        recent = df.iloc[-self.seq_len:]
        if feature_cols is None:
            feature_cols = [c for c in recent.columns
                          if c not in ('label',) and recent[c].dtype in ('float64', 'float32', 'int64')]

        data = recent[feature_cols].values.astype(np.float32)
        # Normalize with full history stats
        full_data = df[feature_cols].values.astype(np.float32)
        for feat in range(data.shape[1]):
            f_mean = full_data[:, feat].mean()
            f_std = full_data[:, feat].std() + 1e-8
            data[:, feat] = (data[:, feat] - f_mean) / f_std

        X = torch.tensor(data).unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            out = model(X)

        return {
            "direction": int(out["direction"].item()),
            "confidence": round(float(out["confidence"].item()), 4),
            "p_up": round(float(out.get("p_up", out["probs"][0, 0]).item()), 4),
            "p_down": round(float(out.get("p_down", out["probs"][0, 1]).item()), 4),
            "p_timeout": round(float(out.get("p_timeout", out["probs"][0, 2]).item()), 4),
        }

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, model: PatchTSTModel, symbol: str, strategy_name: str) -> str:
        path = self.models_dir / f"{symbol}_{strategy_name}_patchtst.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "num_features": model.num_features, "seq_len": model.seq_len,
                "patch_len": model.patch_len, "stride": model.stride,
                "d_model": model.d_model, "num_heads": model.num_heads,
                "num_layers": model.num_layers, "dropout": model.dropout,
            },
        }, path)
        return str(path)

    def load(self, symbol: str, strategy_name: str) -> Optional[PatchTSTModel]:
        path = self.models_dir / f"{symbol}_{strategy_name}_patchtst.pt"
        if not path.exists():
            return None
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt["config"]
        model = PatchTSTModel(
            num_features=cfg["num_features"], seq_len=cfg["seq_len"],
            patch_len=cfg["patch_len"], stride=cfg["stride"],
            d_model=cfg["d_model"], num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        ).to(self.device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
