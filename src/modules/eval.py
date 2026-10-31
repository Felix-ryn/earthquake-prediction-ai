import json
import logging
import math
import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

from src.config import EvalConfig
from src.models.cnn import CNNAngleDir
from src.models.lstm import LSTMAngleDir
from src.modules.train import build_sequence_dataset, split_sequence_indices
from src.utils import setup_logger

logger = setup_logger(__name__)

DIRECTION_ORDER = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTION_ORDER)}

FEATURE_COLS_NB = [
    "center_lat",
    "center_lon",
    "radius_km",
    "area_km2",
    "magnitude",
    "depth_km",
    "hazard_eta",
    "tau_norm",
]


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


def _direction_to_angle(idx_arr: np.ndarray) -> np.ndarray:
    mapping = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype="float32")
    return mapping[idx_arr]


def circular_rmse_deg(true_angle: np.ndarray, pred_angle: np.ndarray) -> float:
    diff = np.asarray(pred_angle, dtype="float64") - np.asarray(
        true_angle, dtype="float64"
    )
    diff = (diff + 180.0) % 360.0 - 180.0
    return math.sqrt(float((diff ** 2).mean()))


def build_nb_dataset(events_path: Path):
    """Dataset per-event untuk evaluasi Naive Bayes (dipertahankan untuk kompatibilitas)."""
    df = _load_events(events_path)

    if "ga_theta_deg" not in df.columns or "ga_direction" not in df.columns:
        raise ValueError("Need ga_theta_deg and ga_direction columns. Run ACO+GA first.")

    df = df.dropna(
        subset=FEATURE_COLS_NB + ["ga_theta_deg", "ga_direction"]
    ).reset_index(drop=True)

    df["is_tectonic"] = (df["type"] == "tectonic").astype(float)
    df["is_volcanic"] = (df["type"] == "volcanic").astype(float)

    feat_cols = FEATURE_COLS_NB + ["is_tectonic", "is_volcanic"]

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[feat_cols].to_numpy(dtype="float32")
    y_dir = df["ga_direction"].map(DIR_TO_IDX).to_numpy(dtype="int64")
    y_angle = df["ga_theta_deg"].to_numpy(dtype="float32")
    return X, y_dir, y_angle


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_angle_true: Optional[np.ndarray] = None,
    y_angle_pred: Optional[np.ndarray] = None,
) -> Dict:
    """Hitung confusion matrix (raw + normalized) dan metrik klasifikasi."""
    labels = list(range(len(DIRECTION_ORDER)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_norm = np.nan_to_num(cm_norm, nan=0.0, posinf=0.0, neginf=0.0)

    acc = accuracy_score(y_true, y_pred)

    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_c, rec_c, f1_c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    per_class = {}
    for idx, d in enumerate(DIRECTION_ORDER):
        per_class[d] = {
            "precision": float(prec_c[idx]),
            "recall": float(rec_c[idx]),
            "f1": float(f1_c[idx]),
            "support": int(support[idx]),
        }

    rmse_angle = float("nan")
    if y_angle_true is not None and y_angle_pred is not None:
        rmse_angle = circular_rmse_deg(
            np.asarray(y_angle_true), np.asarray(y_angle_pred)
        )

    return {
        "accuracy": float(acc),
        "precision_macro": float(prec_m),
        "recall_macro": float(rec_m),
        "f1_macro": float(f1_m),
        "rmse_angle": float(rmse_angle),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_norm.tolist(),
        "support": [int(s) for s in support],
        "per_class": per_class,
    }


def _load_scaler(model_dir: Path, seq_len: int) -> Optional[StandardScaler]:
    scaler_path = Path(model_dir) / f"scaler_seq{seq_len}.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler from %s", scaler_path)
        return scaler
    logger.warning("Scaler %s not found, will fit a fallback scaler.", scaler_path)
    return None


def _evaluate_nb(
    X_raw: np.ndarray,
    y_dir: np.ndarray,
    y_angle: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Dict:
    # NB memakai fitur event terakhir dari sequence, sehingga target event-nya
    # sama dengan target CNN/LSTM (perbandingan apples-to-apples).
    X_train = X_raw[train_idx, -1, :]
    X_test = X_raw[test_idx, -1, :]

    clf = GaussianNB()
    clf.fit(X_train, y_dir[train_idx])

    y_pred = clf.predict(X_test)
    y_angle_pred = _direction_to_angle(y_pred)

    return _metrics_from_predictions(
        y_dir[test_idx], y_pred, y_angle[test_idx], y_angle_pred
    )


def _load_torch_model(model_cls, model_path: Path, n_features: int, device):
    model = model_cls(n_features=n_features).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def _evaluate_torch_model(
    model,
    X_test_scaled: np.ndarray,
    y_dir: np.ndarray,
    y_angle: np.ndarray,
    test_idx: np.ndarray,
    device,
) -> Dict:
    x_tensor = torch.from_numpy(X_test_scaled).to(device)
    pred_angle, logits = model(x_tensor)

    y_pred = logits.argmax(dim=1).cpu().numpy()
    y_angle_pred = pred_angle.cpu().numpy()

    return _metrics_from_predictions(
        y_dir[test_idx], y_pred, y_angle[test_idx], y_angle_pred
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return _json_safe(float(value))
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, data: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", path)


class DirectionEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config

    def run(self) -> Dict:
        cfg = self.config

        X_raw, y_angle, y_dir = build_sequence_dataset(cfg.events_path, cfg.seq_len)

        train_idx, test_idx = split_sequence_indices(
            y_dir, cfg.test_size, cfg.random_state
        )

        n_features = X_raw.shape[-1]
        n_test, T, F = X_raw[test_idx].shape

        scaler = _load_scaler(cfg.model_dir, cfg.seq_len)
        if scaler is None:
            scaler = StandardScaler()
            n_train = X_raw[train_idx].shape[0]
            scaler.fit(X_raw[train_idx].reshape(n_train * T, F))

        X_test_scaled = (
            scaler.transform(X_raw[test_idx].reshape(n_test * T, F))
            .astype("float32")
            .reshape(n_test, T, F)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            "Eval using device=%s, n_train=%d, n_test=%d, seq_len=%d",
            device,
            len(train_idx),
            len(test_idx),
            cfg.seq_len,
        )

        models: Dict[str, Dict] = {}
        skipped: Dict[str, str] = {}

        models["NB"] = _evaluate_nb(X_raw, y_dir, y_angle, train_idx, test_idx)
        logger.info(
            "NB: acc=%.3f, f1_macro=%.3f, rmse_angle=%.2f",
            models["NB"]["accuracy"],
            models["NB"]["f1_macro"],
            models["NB"]["rmse_angle"],
        )

        torch_specs = {
            "CNN": (CNNAngleDir, Path(cfg.model_dir) / "cnn_angle_dir.pt"),
            "LSTM": (LSTMAngleDir, Path(cfg.model_dir) / "lstm_angle_dir.pt"),
        }

        for name, (model_cls, model_path) in torch_specs.items():
            if not model_path.exists():
                logger.warning("%s model not found at %s; skipping.", name, model_path)
                skipped[name] = f"model not found: {model_path}"
                continue
            try:
                model = _load_torch_model(model_cls, model_path, n_features, device)
                models[name] = _evaluate_torch_model(
                    model, X_test_scaled, y_dir, y_angle, test_idx, device
                )
                logger.info(
                    "%s: acc=%.3f, f1_macro=%.3f, rmse_angle=%.2f",
                    name,
                    models[name]["accuracy"],
                    models[name]["f1_macro"],
                    models[name]["rmse_angle"],
                )
            except Exception as exc:
                logger.warning("Failed to evaluate %s (%s); skipping.", name, exc)
                skipped[name] = str(exc)

        combined = {
            "classes": list(DIRECTION_ORDER),
            "meta": {
                "seq_len": int(cfg.seq_len),
                "test_size": float(cfg.test_size),
                "random_state": int(cfg.random_state),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "eval_scope": "shared-sequence-test-set",
                "skipped": skipped,
            },
            "models": models,
        }

        _write_json(cfg.metrics_models_path, combined)
        _write_json(cfg.metrics_web_path, combined)
        _write_json(cfg.metrics_path, models["NB"])

        return combined
