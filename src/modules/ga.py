import json
import logging
import math
from pathlib import Path
import random
from typing import List

import numpy as np
import pandas as pd

from src.config import GAConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


def angle_to_direction(theta_deg: float) -> str:
    d = theta_deg % 360.0
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(((d + 22.5) % 360.0) // 45.0)
    return dirs[idx]


class GAStep:
    def __init__(self, config: GAConfig):
        self.config = config

    def _export_geojson(self, df: pd.DataFrame, path: Path) -> None:
        cfg = self.config
        df = df.copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        
        df_valid = df.dropna(
            subset=["ga_center_lat", "ga_center_lon", "ga_theta_deg", "ga_group_radius_km"]
        )
        
        if "ga_period" in df_valid.columns:
            group_cols = ["type", "ga_period"]
        else:
            group_cols = ["type"]

        reps = (
            df_valid[group_cols + ["ga_center_lat", "ga_center_lon",
                                   "ga_theta_deg", "ga_group_radius_km",
                                   "ga_direction"]]
            .drop_duplicates()
        )

        features = []

        for _, row in reps.iterrows():
            lat0 = float(row["ga_center_lat"])
            lon0 = float(row["ga_center_lon"])
            theta = float(row["ga_theta_deg"])
            step_km = float(row["ga_group_radius_km"])

            theta_rad = math.radians(theta)
            dx = step_km * math.cos(theta_rad)
            dy = step_km * math.sin(theta_rad)

            lat0_rad = math.radians(lat0)
            lat1 = lat0 + dy / 111.0
            lon1 = lon0 + dx / (111.0 * math.cos(lat0_rad))

            props = {
                "type": row.get("type", None),
                "theta_deg": theta,
                "direction": row.get("ga_direction", None),
                "mode": "ga",
                "freq": cfg.freq,
            }
            if "ga_period" in row:
                props["period"] = row["ga_period"]

            feat = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [lon0, lat0],
                        [lon1, lat1],
                    ],
                },
            }
            features.append(feat)

        fc = {"type": "FeatureCollection", "features": features}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fc, f, ensure_ascii=False)
        logger.info("Wrote GA GeoJSON (%d features) to %s", len(features), path)

    def run(self, events_aco: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        df = events_aco.copy().reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        
        if cfg.freq == "M":
            df["ga_period"] = df["datetime"].dt.to_period("M").astype(str)
        elif cfg.freq == "Y":
            df["ga_period"] = df["datetime"].dt.to_period("Y").astype(str)
        else:
            raise ValueError(f"Unsupported GA freq={cfg.freq!r}, use 'M' or 'Y'.")

        df["ga_theta_deg"] = np.nan
        df["ga_direction"] = None
        df["ga_center_lat"] = np.nan
        df["ga_center_lon"] = np.nan
        df["ga_group_radius_km"] = np.nan
        
        for (t, period), group in df.groupby(["type", "ga_period"]):
            idx = group.index.values
            if len(group) < cfg.min_events:
                logger.debug(
                    "Skip GA for type=%s period=%s (n=%d < min_events=%d)",
                    t,
                    period,
                    len(group),
                    cfg.min_events,
                )
                continue
            
            if "hazard_eta_raw" in group.columns:
                w = group["hazard_eta_raw"].to_numpy(dtype="float64")
            else:
                w = group["magnitude"].to_numpy(dtype="float64")
            w = np.maximum(w, 1e-6)
            wsum = w.sum()

            lats = group["latitude"].to_numpy(dtype="float64")
            lons = group["longitude"].to_numpy(dtype="float64")

            lat0 = float(lats.mean())
            lon0 = float(lons.mean())
            lat0_rad = math.radians(lat0)

            xs = (lons - lon0) * 111.0 * math.cos(lat0_rad)
            ys = (lats - lat0) * 111.0
            
            cx = float(np.sum(xs * w) / wsum)
            cy = float(np.sum(ys * w) / wsum)

            xs_c = xs - cx
            ys_c = ys - cy

            def fitness(theta_rad: float) -> float:
                ux = math.cos(theta_rad)
                uy = math.sin(theta_rad)
                proj = xs_c * ux + ys_c * uy
                mean_proj = float(np.sum(w * proj) / wsum)
                var = float(np.sum(w * (proj - mean_proj) ** 2) / wsum)
                return var
          
            pop = np.random.uniform(0.0, 2.0 * math.pi, size=cfg.pop_size)
            elite_size = max(1, int(cfg.pop_size * cfg.elite_frac))

            for gen in range(cfg.n_generations):
                fitness_vals = np.array([fitness(th) for th in pop])
                order = np.argsort(-fitness_vals)  
                pop = pop[order]
                fitness_vals = fitness_vals[order]

                elites = pop[:elite_size]
                new_pop = [float(th) for th in elites]

                
                while len(new_pop) < cfg.pop_size:
                    i, j = np.random.randint(0, elite_size, size=2)
                    p1, p2 = pop[i], pop[j]
                    child = 0.5 * (p1 + p2)

                    
                    std_rad = math.radians(cfg.mutation_std_deg) * (
                        0.5 + 0.5 * (1.0 - gen / max(1, cfg.n_generations))
                    )
                    child += np.random.normal(0.0, std_rad)
                    child = child % (2.0 * math.pi)

                    new_pop.append(child)

                pop = np.array(new_pop)
            
            fitness_vals = np.array([fitness(th) for th in pop])
            best_idx = int(np.argmax(fitness_vals))
            best_theta = float(pop[best_idx])      
            best_theta_deg = math.degrees(best_theta)
            direction_label = angle_to_direction(best_theta_deg)
            
            ga_center_lat = lat0 + (cy / 111.0)
            ga_center_lon = lon0 + (cx / (111.0 * math.cos(lat0_rad)))

            dists = np.hypot(xs_c, ys_c)
            rms_dist = math.sqrt(float(np.sum(w * dists**2) / wsum))

            df.loc[idx, "ga_theta_deg"] = best_theta_deg
            df.loc[idx, "ga_direction"] = direction_label
            df.loc[idx, "ga_center_lat"] = ga_center_lat
            df.loc[idx, "ga_center_lon"] = ga_center_lon
            df.loc[idx, "ga_group_radius_km"] = rms_dist

            logger.info(
                "GA(type=%s, period=%s): theta=%.2f deg, dir=%s, R=%.2f km, n=%d",
                t,
                period,
                best_theta_deg,
                direction_label,
                rms_dist,
                len(group),
            )

        if cfg.geojson_path is not None:
            self._export_geojson(df, cfg.geojson_path)

        return df
