import json
import logging
import math
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.config import ACOConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


def compute_hazard_eta(row: pd.Series) -> float:
    mag = float(row["magnitude"])
    depth = max(float(row["depth_km"]), 0.1)
    quake_type = row.get("type", "")

    if quake_type == "volcanic":
        frp_val = row.get("frp_mw", np.nan)
        frp = float(frp_val) if not pd.isna(frp_val) else 0.1
        dist_val = row.get("hotspot_distance_km", np.nan)
        dist = float(dist_val) if not pd.isna(dist_val) else 10.0
        eta = (mag ** 1.5) * math.log1p(frp) / (depth ** 0.5 * math.log1p(dist))
    else:
        eta = (mag ** 2.0) / (depth ** 0.5)

    return float(eta)


class ACOStep:
    def __init__(self, config: ACOConfig):
        self.config = config

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["_dt"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        df = df.sort_values("_dt").reset_index(drop=True)
        return df

    def _build_neighbors(self, n: int, k: int) -> List[list]:
        neighbors: List[list] = []
        for i in range(n):
            start = max(0, i - k)
            end = min(n, i + k + 1)
            idxs = [j for j in range(start, end) if j != i]
            neighbors.append(idxs)
        return neighbors

    def _export_geojson(self, df: pd.DataFrame, path: Path) -> None:
        features = []
        for _, row in df.iterrows():
            if pd.isna(row.get("center_lat")) or pd.isna(row.get("center_lon")):
                continue

            props = {
                "datetime": row["datetime"].isoformat() if not pd.isna(row["datetime"]) else None,
                "magnitude": float(row["magnitude"]) if not pd.isna(row.get("magnitude")) else None,
                "depth_km": float(row["depth_km"]) if not pd.isna(row.get("depth_km")) else None,
                "type": row.get("type", None),
                "hazard_eta": float(row["hazard_eta"]) if not pd.isna(row.get("hazard_eta")) else None,
                "tau_norm": float(row["tau_norm"]) if not pd.isna(row.get("tau_norm")) else None,
                "radius_km": float(row["radius_km"]) if not pd.isna(row.get("radius_km")) else None,
                "area_km2": float(row["area_km2"]) if not pd.isna(row.get("area_km2")) else None,
            }

            feat = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row["center_lon"]), float(row["center_lat"])],
                },
            }
            features.append(feat)

        fc = {"type": "FeatureCollection", "features": features}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        logger.info("Wrote ACO GeoJSON (%d features) to %s", len(features), path)

    def run(self, events: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        logger.info(
            "Running ACO on %d events (n_ants=%d, n_iters=%d)",
            len(events),
            cfg.n_ants,
            cfg.n_iters,
        )

        df = self._prepare(events)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        hazard_raw = df.apply(compute_hazard_eta, axis=1).values
        eta_min, eta_max = hazard_raw.min(), hazard_raw.max()
        if eta_max <= eta_min:
            eta_norm = np.ones_like(hazard_raw)
        else:
            eta_norm = (hazard_raw - eta_min) / (eta_max - eta_min) + 1e-6

        n = len(df)

        neighbors = self._build_neighbors(n, cfg.neighbor_k)

        tau = np.ones(n, dtype="float64")
        Q = 1.0

        times = df["_dt"].map(
            lambda ts: ts.timestamp() if not pd.isna(ts) else float("nan")
        ).to_numpy()
        lats = df["latitude"].to_numpy(dtype="float64")
        lons = df["longitude"].to_numpy(dtype="float64")

        def edge_eta(i: int, j: int) -> float:
            dt_days = abs(times[j] - times[i]) / (3600.0 * 24.0)
            lat_mid = (lats[i] + lats[j]) / 2.0
            lat_mid_rad = math.radians(lat_mid)
            dx = (lons[j] - lons[i]) * 111.0 * math.cos(lat_mid_rad)
            dy = (lats[j] - lats[i]) * 111.0
            dist_km = math.hypot(dx, dy)
            return 1.0 / (1.0 + dt_days + dist_km / 100.0)

        edge_cache: dict[tuple, float] = {}
        for i in range(n):
            for j in neighbors[i]:
                edge_cache[(i, j)] = edge_eta(i, j)

        start_probs = eta_norm / eta_norm.sum()
        for _ in range(cfg.n_iters):
            delta_tau = np.zeros(n, dtype="float64")

            for _ant in range(cfg.n_ants):
                current = np.random.choice(n, p=start_probs)

                for _step in range(cfg.walk_length):
                    neigh = neighbors[current]
                    if not neigh:
                        break

                    scores = []
                    for j in neigh:
                        tau_j = tau[j] ** cfg.alpha
                        eta_ij = edge_cache[(current, j)] ** cfg.beta
                        scores.append(tau_j * eta_ij)

                    total = sum(scores)
                    if total <= 0:
                        break

                    probs = [s / total for s in scores]
                    idx = np.random.choice(len(neigh), p=probs)
                    nxt = neigh[idx]

                    delta_tau[nxt] += Q * eta_norm[nxt]
                    current = nxt

            tau = (1.0 - cfg.rho) * tau + delta_tau

        tau_min, tau_max = tau.min(), tau.max()
        if tau_max <= tau_min:
            tau_norm = np.ones_like(tau)
        else:
            tau_norm = (tau - tau_min) / (tau_max - tau_min)

        radii_km = []
        for i, row in df.iterrows():
            mag = float(row["magnitude"])
            depth = max(float(row["depth_km"]), 0.1)
            t = row.get("type", "")

            if t == "volcanic":
                frp_val = row.get("frp_mw", np.nan)
                frp = float(frp_val) if not pd.isna(frp_val) else 0.1
                r0 = 5.0 + 2.0 * (mag - 3.0) + 3.0 * math.log1p(frp)
                r0 = max(3.0, min(r0, 50.0))
                min_r = 3.0
            else:
                rupture = 10 ** (0.5 * mag - 1.8)
                r0 = rupture * 1.5
                r0 = max(5.0, min(r0, 120.0))
                min_r = 1.0

            scale = 0.75 + 0.5 * tau_norm[i]
            r = max(min_r, r0 * scale)
            radii_km.append(r)

        radii_km = np.asarray(radii_km, dtype="float64")
        areas_km2 = math.pi * radii_km ** 2

        df["hazard_eta_raw"] = hazard_raw
        df["hazard_eta"] = eta_norm
        df["tau"] = tau
        df["tau_norm"] = tau_norm
        df["radius_km"] = radii_km
        df["area_km2"] = areas_km2
        df["center_lat"] = df["latitude"]
        df["center_lon"] = df["longitude"]

        df = df.drop(columns=["_dt"])

        if cfg.geojson_path is not None:
            self._export_geojson(df, cfg.geojson_path)

        return df
