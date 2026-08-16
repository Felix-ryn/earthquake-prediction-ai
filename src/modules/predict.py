import json
import logging
import math
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.config import PredictConfig
from src.modules.train import FEATURE_COLS, DIRECTION_ORDER, _load_events
from src.models.lstm import LSTMAngleDir
from src.models.cnn import CNNAngleDir
from src.utils import setup_logger

logger = setup_logger(__name__)


def _safe_float(val):
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    try:
        return float(val)
    except Exception:
        return None


def _load_scaler(model_dir: Path, seq_len: int) -> StandardScaler | None:
    model_dir = Path(model_dir)
    scaler_path = model_dir / f"scaler_seq{seq_len}.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        logger.info("Loaded scaler from %s", scaler_path)
        return scaler
    logger.warning("Scaler %s not found, will be refitted from data if needed.", scaler_path)
    return None


def build_latest_sequence(
    events_path: Path,
    seq_len: int,
    scaler: StandardScaler | None,
) -> Tuple[np.ndarray, pd.Series, pd.DataFrame, StandardScaler]:
    df = _load_events(events_path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df = df.sort_values("datetime").reset_index(drop=True)

    df = df.dropna(
        subset=[
            "center_lat",
            "center_lon",
            "radius_km",
            "area_km2",
        ]
    ).reset_index(drop=True)

    if len(df) < seq_len:
        raise ValueError(
            f"Dataset only has {len(df)} valid events, but seq_len={seq_len}."
        )

    df["is_tectonic"] = (df["type"] == "tectonic").astype(float)
    df["is_volcanic"] = (df["type"] == "volcanic").astype(float)

    feat_cols = FEATURE_COLS + ["is_tectonic", "is_volcanic"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    last_type = df.iloc[-1].get("type")
    df_type = df[df["type"] == last_type].copy()
    df_type = df_type.sort_values("datetime").reset_index(drop=True)

    if len(df_type) >= seq_len:
        df_seq = df_type.iloc[-seq_len:].copy().reset_index(drop=True)
        logger.info(
            "Building latest sequence from type=%s (n=%d, seq_len=%d).",
            last_type,
            len(df_type),
            seq_len,
        )
    else:
        logger.warning(
            "Not enough events for type=%s (n=%d < seq_len=%d); "
            "using last %d events from all types instead.",
            last_type,
            len(df_type),
            seq_len,
            seq_len,
        )
        df_seq = df.iloc[-seq_len:].copy().reset_index(drop=True)

    X_raw = df_seq[feat_cols].to_numpy(dtype="float32")

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_raw)
        logger.info("Fitted new scaler on latest sequence (fallback, no saved scaler).")

    X_scaled = scaler.transform(X_raw).astype("float32")
    X_seq = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])

    meta_last = df_seq.iloc[-1]

    logger.info(
        "Built latest sequence (seq_len=%d). Last datetime=%s, type=%s",
        seq_len,
        meta_last["datetime"],
        meta_last.get("type"),
    )

    return X_seq, meta_last, df_seq, scaler


def forecast_roi(
    meta_last: pd.Series,
    pred_angle_deg: float,
    step_km: float | None = None,
    roi_radius_km: float | None = None,
    n_vertices: int = 36,
) -> dict:
    """
    Dari sudut prediksi + pusat GA terakhir → pusat ROI baru + poligon ROI.
    0° = timur, 90° = utara, 180° = barat, 270° = selatan.
    """

    def _field(s: pd.Series, *names: str) -> float:
        for name in names:
            if name in s.index and not pd.isna(s[name]):
                return float(s[name])
        raise ValueError(f"No valid numeric value found for {names}")

    lat0 = _field(meta_last, "win_center_lat", "center_lat")
    lon0 = _field(meta_last, "win_center_lon", "center_lon")

    if step_km is None:
        if "ga_group_radius_km" in meta_last and not pd.isna(meta_last["ga_group_radius_km"]):
            step_km = float(meta_last["ga_group_radius_km"])
        else:
            step_km = float(meta_last["radius_km"])


    if roi_radius_km is None:
        roi_radius_km = float(meta_last["radius_km"])

    theta_rad = math.radians(pred_angle_deg)
    dx = step_km * math.cos(theta_rad)
    dy = step_km * math.sin(theta_rad)

    lat0_rad = math.radians(lat0)
    pred_lat = lat0 + dy / 111.0
    pred_lon = lon0 + dx / (111.0 * math.cos(lat0_rad))

    coords = []
    for i in range(n_vertices):
        phi_deg = (360.0 * i) / n_vertices
        phi = math.radians(phi_deg)
        dx_r = roi_radius_km * math.cos(phi)
        dy_r = roi_radius_km * math.sin(phi)

        lat = pred_lat + dy_r / 111.0
        lon = pred_lon + dx_r / (111.0 * math.cos(math.radians(pred_lat)))
        coords.append([lon, lat])
    coords.append(coords[0])

    return {
        "pred_center_lat": pred_lat,
        "pred_center_lon": pred_lon,
        "roi_radius_km": roi_radius_km,
        "step_km": step_km,
        "polygon_coords": coords,
    }


class Predictor:
    def __init__(self, config: PredictConfig):
        self.config = config

    def run(self) -> dict:
        cfg = self.config
        scaler = _load_scaler(cfg.model_dir, cfg.seq_len)

        X_seq, meta_last, df_seq, scaler = build_latest_sequence(
            cfg.events_path, cfg.seq_len, scaler
        )

        model_path = Path(cfg.model_dir) / "lstm_angle_dir.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found, train LSTM first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_features = X_seq.shape[-1]

        model = LSTMAngleDir(n_features=n_features).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        x_tensor = torch.from_numpy(X_seq).to(device)
        with torch.no_grad():
            pred_angle_tensor, logits = model(x_tensor)

        pred_angle_raw = float(pred_angle_tensor.cpu().numpy()[0])
        pred_angle_deg = (pred_angle_raw % 360.0 + 360.0) % 360.0

        pred_dir_idx = int(torch.argmax(logits, dim=1).cpu().numpy()[0])
        pred_dir_label = DIRECTION_ORDER[pred_dir_idx]

        logger.info(
            "Predicted angle_raw=%.2f deg, angle_norm=%.2f deg, direction=%s",
            pred_angle_raw,
            pred_angle_deg,
            pred_dir_label,
        )

        roi_info = forecast_roi(meta_last, pred_angle_deg)

        last_dt = pd.to_datetime(meta_last.get("datetime"), utc=True, errors="coerce")
        try:
            last_dt_iso = last_dt.isoformat()
        except Exception:
            last_dt_iso = None

        raw_lat = meta_last.get("win_center_lat")
        if raw_lat is None or pd.isna(raw_lat):
            raw_lat = meta_last.get("center_lat")
        last_center_lat = _safe_float(raw_lat)

        raw_lon = meta_last.get("win_center_lon")
        if raw_lon is None or pd.isna(raw_lon):
            raw_lon = meta_last.get("center_lon")
        last_center_lon = _safe_float(raw_lon)

        props = {
            "model": "LSTM",
            "seq_len": cfg.seq_len,
            "pred_theta_raw_deg": pred_angle_raw,
            "pred_theta_deg": pred_angle_deg,
            "pred_direction": pred_dir_label,
            "pred_step_km": roi_info["step_km"],
            "pred_roi_radius_km": roi_info["roi_radius_km"],
            "pred_center_lat": roi_info["pred_center_lat"],
            "pred_center_lon": roi_info["pred_center_lon"],
            "last_datetime": last_dt_iso,
            "last_center_lat": last_center_lat,
            "last_center_lon": last_center_lon,
            "last_type": meta_last.get("type"),
            "last_magnitude": _safe_float(meta_last.get("magnitude")),
            "last_depth_km": _safe_float(meta_last.get("depth_km")),
            "last_hazard_eta": _safe_float(meta_last.get("hazard_eta")),
            "last_tau_norm": _safe_float(meta_last.get("tau_norm")),
            "last_radius_km": _safe_float(meta_last.get("radius_km")),
            "n_sequence_events": int(len(df_seq)),
            "source_step": "predict_roi",
        }

        feature = {
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Polygon",
                "coordinates": [roi_info["polygon_coords"]],
            },
        }
        geojson = {"type": "FeatureCollection", "features": [feature]}

        roi_geojson = cfg.roi_geojson_path
        roi_geojson.parent.mkdir(parents=True, exist_ok=True)
        with open(roi_geojson, "w", encoding="utf-8") as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)
        logger.info("Saved ROI GeoJSON to %s", roi_geojson)

        summary = {
            "pred_angle_deg": pred_angle_deg,
            "pred_direction": pred_dir_label,
            "roi_info": roi_info,
            "geojson_path": str(roi_geojson),
        }

        return summary

