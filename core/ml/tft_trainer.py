"""TFT trainer — walk-forward sequence preparation and training loop."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional
from loguru import logger

from core.ml.tft_model import TFTModel, pinball_loss
from core.ml.features import compute_features, REQUIRED_INDICATORS


class TFTTrainer:
    """Sequence-aware TFT trainer with walk-forward data preparation.

    Converts flat OHLCV DataFrames into (seq_len, features) sliding windows.
    Training uses pinball loss for multi-quantile prediction.
    """

    def __init__(self,
                 data_dir: str,
                 seq_len: int = 100,
                 d_model: int = 64,
                 num_heads: int = 4,
                 lstm_layers: int = 2,
                 dropout: float = 0.2,
                 device: str | None = None):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.seq_len = seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.lstm_layers = lstm_layers
        self.dropout = dropout

        if device is None:
            if torch.cuda.is_available():
                try:
                    # Test CUDA with a tiny op — RTX 50xx needs CUDA 13+
                    _test = torch.zeros(1).cuda()
                    self.device = "cuda"
                except Exception:
                    self.device = "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"TFT device: {self.device}")

    # ── Data Preparation ─────────────────────────────────────────────

    def prepare_sequences(self, df: pd.DataFrame,
                          feature_cols: list[str] | None = None,
                          label_col: str = "label") -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a flat feature DataFrame into (X_seq, y) tensors.

        Each sample: features from t-seq_len+1 to t, label at t.

        Parameters
        ----------
        df : pd.DataFrame
            Must have all feature columns + a 'label' column.
            We drop the last few rows to avoid NaN labels.

        Returns
        -------
        X : (num_sequences, seq_len, num_features)
        y : (num_sequences,) — forward return labels
        """
        if feature_cols is None:
            # Auto-detect: all numeric columns except 'label'
            feature_cols = [c for c in df.columns
                          if c != label_col and df[c].dtype in ('float64', 'float32', 'int64')]

        data = df[feature_cols].values.astype(np.float32)
        labels = df[label_col].values.astype(np.float32)

        # Drop rows with NaN in features or labels
        valid_mask = ~(np.isnan(data).any(axis=1) | np.isnan(labels))
        data = data[valid_mask]
        labels = labels[valid_mask]

        if len(data) <= self.seq_len:
            return (torch.empty(0, self.seq_len, len(feature_cols)),
                    torch.empty(0))

        # Build sliding windows
        num_samples = len(data) - self.seq_len
        X_list = []
        y_list = []
        for i in range(num_samples):
            X_list.append(data[i:i + self.seq_len])
            y_list.append(labels[i + self.seq_len])

        if not X_list:
            return (torch.empty(0, self.seq_len, len(feature_cols)),
                    torch.empty(0))

        X = np.stack(X_list, axis=0)
        y = np.array(y_list)

        # Standardize with expanding-window stats (walk-forward safe).
        # For each window at position i, use mean/std of all data [0 : i+seq_len].
        # This preserves trend info — model can see if current value is high/low
        # relative to history, not just relative to the last 100 bars.
        for feat in range(X.shape[2]):
            feat_data = data[:num_samples + self.seq_len, feat]
            cumsum = np.cumsum(feat_data)
            cumsum2 = np.cumsum(feat_data ** 2)
            for i in range(num_samples):
                end = i + self.seq_len  # last index in this window
                n = end + 1
                f_mean = cumsum[end] / n
                f_var = cumsum2[end] / n - f_mean ** 2
                f_std = np.sqrt(max(f_var, 1e-10)) + 1e-8
                X[i, :, feat] = (X[i, :, feat] - f_mean) / f_std

        return torch.tensor(X), torch.tensor(y)

    # ── Training ─────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame,
              feature_cols: list[str] | None = None,
              label_col: str = "label",
              epochs: int = 80,
              batch_size: int = 64,
              learning_rate: float = 1e-3,
              validation_split: float = 0.2,
              patience: int = 20) -> tuple[Optional[TFTModel], dict]:
        """Train a TFT model on *df*.

        Returns (model, metrics_dict).
        """
        X, y = self.prepare_sequences(df, feature_cols, label_col)
        if len(X) < 60:
            return None, {"error": f"Insufficient sequences: {len(X)}"}

        # Train/val split (temporal — no shuffle!)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        num_features = X.shape[2]
        model = TFTModel(
            num_features=num_features,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=self.num_heads,
            lstm_layers=self.lstm_layers,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False)

        for epoch in range(epochs):
            # ── Train ──
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                out = model(batch_X)
                loss = pinball_loss(out['quantiles'], batch_y, model.quantiles)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= max(len(train_loader), 1)

            # ── Validate ──
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    out = model(batch_X)
                    loss = pinball_loss(out['quantiles'], batch_y, model.quantiles)
                    val_loss += loss.item()

                    # Direction accuracy (P50 sign vs true sign)
                    true_dir = torch.sign(batch_y)
                    pred_dir = out['direction']
                    val_correct += (true_dir == pred_dir).sum().item()
                    val_total += len(batch_y)

            val_loss /= max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.debug(f"TFT epoch {epoch+1}/{epochs}: "
                           f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                           f"val_acc={val_acc:.3f}")

            if patience_counter >= patience:
                logger.debug(f"TFT early stop at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = {
            "val_loss": best_val_loss,
            "val_accuracy": val_acc,
            "epochs_trained": epoch + 1,
            "num_sequences": len(X),
            "num_features": num_features,
            "device": self.device,
        }
        return model, metrics

    # ── Prediction ───────────────────────────────────────────────────

    def predict(self, model: TFTModel,
                df: pd.DataFrame,
                feature_cols: list[str] | None = None) -> dict | None:
        """Make a single prediction from the latest sequence window.

        Parameters
        ----------
        model : TFTModel
            Trained model.
        df : pd.DataFrame
            Feature DataFrame (must have ≥ seq_len rows).
        feature_cols : list[str] | None
            Feature columns to use.

        Returns
        -------
        dict with keys: direction, confidence, uncertainty, p50, quantiles
        """
        if len(df) < self.seq_len:
            return None

        # Get the last seq_len rows
        recent = df.iloc[-self.seq_len:]

        if feature_cols is None:
            feature_cols = [c for c in recent.columns
                          if c not in ('label',) and recent[c].dtype in ('float64', 'float32', 'int64')]

        data = recent[feature_cols].values.astype(np.float32)

        # Standardize using this window's statistics
        for feat in range(data.shape[1]):
            f_mean = data[:, feat].mean()
            f_std = data[:, feat].std() + 1e-8
            data[:, feat] = (data[:, feat] - f_mean) / f_std

        X = torch.tensor(data).unsqueeze(0).to(self.device)  # (1, seq, features)

        model.eval()
        with torch.no_grad():
            out = model(X)

        return {
            "direction": int(out["direction"].item()),
            "confidence": round(float(out["confidence"].item()), 4),
            "uncertainty": round(float(out["uncertainty"].item()), 6),
            "p50": round(float(out["p50"].item()), 6),
            "quantiles": [round(float(q), 6) for q in out["quantiles"][0].tolist()],
        }

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, model: TFTModel, symbol: str, strategy_name: str) -> str:
        path = self.models_dir / f"{symbol}_{strategy_name}_tft.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "num_features": model.num_features,
                "seq_len": model.seq_len,
                "d_model": model.d_model,
                "num_heads": model.num_heads,
                "lstm_layers": model.lstm_layers,
                "dropout": model.dropout,
                "quantiles": model.quantiles,
            },
        }, path)
        return str(path)

    def load(self, symbol: str, strategy_name: str) -> Optional[TFTModel]:
        path = self.models_dir / f"{symbol}_{strategy_name}_tft.pt"
        if not path.exists():
            return None
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        cfg = checkpoint["config"]
        model = TFTModel(
            num_features=cfg["num_features"],
            seq_len=cfg["seq_len"],
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
            quantiles=cfg["quantiles"],
        ).to(self.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
