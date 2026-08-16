import json
import logging
import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from src.config import EvalConfig
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


def build_nb_dataset(events_path: Path):
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


def circular_rmse_deg(true_angle: np.ndarray, pred_angle: np.ndarray) -> float:
    diff = pred_angle - true_angle
    diff = (diff + 180.0) % 360.0 - 180.0
    return math.sqrt(float((diff ** 2).mean()))


class DirectionEvaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.model_: GaussianNB | None = None

    def run(self) -> Dict:
        cfg = self.config
        X, y_dir, y_angle = build_nb_dataset(cfg.events_path)

        stratify_param = y_dir if cfg.stratify else None
        (
            X_train,
            X_test,
            y_dir_train,
            y_dir_test,
            y_angle_train,
            y_angle_test,
        ) = train_test_split(
            X,
            y_dir,
            y_angle,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
            stratify=stratify_param,
        )

        clf = GaussianNB()
        clf.fit(X_train, y_dir_train)
        self.model_ = clf

        y_dir_pred = clf.predict(X_test)
        acc = accuracy_score(y_dir_test, y_dir_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_dir_test, y_dir_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_dir_test, y_dir_pred)
        
        y_angle_pred = _direction_to_angle(y_dir_pred)        
        rmse_angle = circular_rmse_deg(y_angle_test, y_angle_pred)

        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
            y_dir_test,
            y_dir_pred,
            labels=list(range(len(DIRECTION_ORDER))),
            zero_division=0,
        )

        per_class = {}
        for idx, d in enumerate(DIRECTION_ORDER):
            per_class[d] = {
                "precision": float(prec_c[idx]),
                "recall": float(rec_c[idx]),
                "f1": float(f1_c[idx]),
            }

        metrics = {
            "accuracy": float(acc),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
            "rmse_angle": float(rmse_angle),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class,
        }

        logger.info(
            "Naive Bayes: acc=%.3f, prec_m=%.3f, rec_m=%.3f, f1_m=%.3f, rmse_angle=%.2f",
            acc,
            prec_m,
            rec_m,
            f1_m,
            rmse_angle,
        )

        metrics_path = cfg.metrics_path
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved NB metrics to %s", metrics_path)

        return metrics
