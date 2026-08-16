import logging
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import TrainConfig
from src.models.cnn import CNNAngleDir
from src.models.lstm import LSTMAngleDir
from src.utils import setup_logger

logger = setup_logger(__name__)

FEATURE_COLS = [
    "center_lat",
    "center_lon",
    "radius_km",
    "area_km2",
    "magnitude",
    "depth_km",
    "hazard_eta",
    "tau_norm",
]

DIRECTION_ORDER = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTION_ORDER)}


def _load_events(events_path: Path) -> pd.DataFrame:
    events_path = Path(events_path)
    if events_path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(events_path, engine="fastparquet")
        except Exception:
            csv_path = events_path.with_suffix(".csv")
            df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
    else:
        df = pd.read_csv(events_path, encoding="utf-8", on_bad_lines="skip")
    return df


def build_sequence_dataset(events_path: Path, seq_len: int = 10):
    df = _load_events(events_path)

    if "ga_theta_deg" not in df.columns or "ga_direction" not in df.columns:
        raise ValueError(
            "Dataset must contain ga_theta_deg and ga_direction. "
            "Run ACO+GA pipeline first."
        )

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.dropna(
        subset=[
            "center_lat",
            "center_lon",
            "radius_km",
            "area_km2",
            "ga_theta_deg",
            "ga_direction",
        ]
    )

    df["is_tectonic"] = (df["type"] == "tectonic").astype(float)
    df["is_volcanic"] = (df["type"] == "volcanic").astype(float)

    feat_cols = FEATURE_COLS + ["is_tectonic", "is_volcanic"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataset: {missing}")

    X_seqs = []
    y_angle = []
    y_dir = []

    for t, group in df.groupby("type"):
        if len(group) < seq_len:
            continue

        group = group.sort_values("datetime")
        feats = group[feat_cols].to_numpy(dtype="float32")
        angles = group["ga_theta_deg"].to_numpy(dtype="float32")
        dirs = group["ga_direction"].map(DIR_TO_IDX).to_numpy(dtype="int64")

        for i in range(seq_len - 1, len(group)):
            seq = feats[i - seq_len + 1 : i + 1]
            if np.any(np.isnan(seq)):
                continue
            X_seqs.append(seq)
            y_angle.append(angles[i])
            y_dir.append(dirs[i])

    if not X_seqs:
        raise ValueError("No valid sequences could be built; check data and seq_len.")

    X_seqs = np.stack(X_seqs).astype("float32")
    y_angle = np.array(y_angle, dtype="float32")
    y_dir = np.array(y_dir, dtype="int64")

    N, T, F = X_seqs.shape
    logger.info(
        "Built (UNSCALED) sequence dataset from %s: N=%d, seq_len=%d, n_features=%d",
        events_path,
        N,
        T,
        F,
    )

    return X_seqs, y_angle, y_dir


def _make_loader(
    X: np.ndarray,
    y_angle: np.ndarray,
    y_dir: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    tensor_x = torch.from_numpy(X)
    tensor_angle = torch.from_numpy(y_angle)
    tensor_dir = torch.from_numpy(y_dir)
    ds = TensorDataset(tensor_x, tensor_angle, tensor_dir)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def angular_loss_deg(pred_angle: torch.Tensor, true_angle: torch.Tensor) -> torch.Tensor:
    diff_rad = (pred_angle - true_angle) * math.pi / 180.0
    return torch.mean(1.0 - torch.cos(diff_rad))


def _train_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_reg: float,
    lambda_cls: float,
):
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    for xb, y_angle, y_dir in loader:
        xb = xb.to(device)
        y_angle = y_angle.to(device)
        y_dir = y_dir.to(device)

        optimizer.zero_grad()
        pred_angle, logits = model(xb)

        loss_reg = angular_loss_deg(pred_angle, y_angle)
        loss_cls = ce(logits, y_dir)
        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)

    if total_n == 0:
        return float("nan")
    return total_loss / total_n


@torch.no_grad()
def _eval_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
    lambda_reg: float,
    lambda_cls: float,
):
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    all_true_dir = []
    all_pred_dir = []
    all_true_angle = []
    all_pred_angle = []

    for xb, y_angle, y_dir in loader:
        xb = xb.to(device)
        y_angle = y_angle.to(device)
        y_dir = y_dir.to(device)

        pred_angle, logits = model(xb)

        loss_reg = angular_loss_deg(pred_angle, y_angle)
        loss_cls = ce(logits, y_dir)
        loss = lambda_reg * loss_reg + lambda_cls * loss_cls

        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)

        preds = logits.argmax(dim=1).cpu().numpy()
        all_true_dir.append(y_dir.cpu().numpy())
        all_pred_dir.append(preds)
        all_true_angle.append(y_angle.cpu().numpy())
        all_pred_angle.append(pred_angle.cpu().numpy())

    if total_n == 0:
        return {"loss": float("nan"), "rmse_angle": float("nan"), "acc_dir": float("nan")}

    all_true_dir = np.concatenate(all_true_dir)
    all_pred_dir = np.concatenate(all_pred_dir)
    all_true_angle = np.concatenate(all_true_angle)
    all_pred_angle = np.concatenate(all_pred_angle)

    diff = all_pred_angle - all_true_angle
    diff = (diff + 180.0) % 360.0 - 180.0
    rmse_angle = math.sqrt(float((diff ** 2).mean()))

    acc_dir = float((all_true_dir == all_pred_dir).mean())

    return {"loss": total_loss / total_n, "rmse_angle": rmse_angle, "acc_dir": acc_dir}


class Trainer:
    def __init__(self, config: TrainConfig):
        self.config = config

    def train(self) -> dict:
        cfg = self.config
        model_dir = Path(cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        X_raw, y_angle, y_dir = build_sequence_dataset(cfg.events_path, cfg.seq_len)

        (
            X_train_raw,
            X_val_raw,
            ya_train,
            ya_val,
            yd_train,
            yd_val,
        ) = train_test_split(
            X_raw,
            y_angle,
            y_dir,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=y_dir,
        )

        N_train, T, F = X_train_raw.shape
        scaler = StandardScaler()
        X_train_flat = X_train_raw.reshape(N_train * T, F)
        scaler.fit(X_train_flat)

        X_train_scaled = (
            scaler.transform(X_train_flat).astype("float32").reshape(N_train, T, F)
        )

        N_val = X_val_raw.shape[0]
        X_val_flat = X_val_raw.reshape(N_val * T, F)
        X_val_scaled = (
            scaler.transform(X_val_flat).astype("float32").reshape(N_val, T, F)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        train_loader = _make_loader(X_train_scaled, ya_train, yd_train, cfg.batch_size, True)
        val_loader = _make_loader(X_val_scaled, ya_val, yd_val, cfg.batch_size, False)

        n_features = X_raw.shape[-1]

        lambda_reg = getattr(cfg, "lambda_reg", 0.5)
        lambda_cls = getattr(cfg, "lambda_cls", 0.5)

        cnn = CNNAngleDir(n_features=n_features).to(device)
        opt_cnn = torch.optim.Adam(cnn.parameters(), lr=cfg.learning_rate)
        best_cnn_state = None
        best_cnn_val_loss = float("inf")

        for epoch in range(1, cfg.epochs_cnn + 1):
            train_loss = _train_epoch(
                cnn, train_loader, opt_cnn, device, lambda_reg, lambda_cls
            )
            val_metrics = _eval_epoch(
                cnn, val_loader, device, lambda_reg, lambda_cls
            )
            val_loss = val_metrics["loss"]
            logger.info(
                "[CNN] epoch=%d train_loss=%.4f val_loss=%.4f "
                "val_acc_dir=%.3f val_rmse_angle=%.2f",
                epoch,
                train_loss,
                val_loss,
                val_metrics["acc_dir"],
                val_metrics["rmse_angle"],
            )
            if val_loss < best_cnn_val_loss:
                best_cnn_val_loss = val_loss
                best_cnn_state = cnn.state_dict()

        cnn_path = model_dir / "cnn_angle_dir.pt"
        if best_cnn_state is not None:
            torch.save(best_cnn_state, cnn_path)
            logger.info("Saved best CNN model to %s", cnn_path)

        lstm = LSTMAngleDir(n_features=n_features).to(device)
        opt_lstm = torch.optim.Adam(lstm.parameters(), lr=cfg.learning_rate)
        best_lstm_state = None
        best_lstm_val_loss = float("inf")

        for epoch in range(1, cfg.epochs_lstm + 1):
            train_loss = _train_epoch(
                lstm, train_loader, opt_lstm, device, lambda_reg, lambda_cls
            )
            val_metrics = _eval_epoch(
                lstm, val_loader, device, lambda_reg, lambda_cls
            )
            val_loss = val_metrics["loss"]
            logger.info(
                "[LSTM] epoch=%d train_loss=%.4f val_loss=%.4f "
                "val_acc_dir=%.3f val_rmse_angle=%.2f",
                epoch,
                train_loss,
                val_loss,
                val_metrics["acc_dir"],
                val_metrics["rmse_angle"],
            )
            if val_loss < best_lstm_val_loss:
                best_lstm_val_loss = val_loss
                best_lstm_state = lstm.state_dict()

        lstm_path = model_dir / "lstm_angle_dir.pt"
        if best_lstm_state is not None:
            torch.save(best_lstm_state, lstm_path)
            logger.info("Saved best LSTM model to %s", lstm_path)

        scaler_path = model_dir / f"scaler_seq{cfg.seq_len}.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logger.info("Saved scaler to %s", scaler_path)

        return {
            "cnn_model_path": cnn_path,
            "lstm_model_path": lstm_path,
            "scaler_path": scaler_path,
        }
