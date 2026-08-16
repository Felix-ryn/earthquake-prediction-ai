import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import IngestConfig
from src.utils import setup_logger

logger = setup_logger(__name__)


class EventIngestor:
    def __init__(self, config: IngestConfig):
        self.config = config

    def _safe_datetime(self, df: pd.DataFrame) -> pd.Series:
        dt = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
        bad = df[dt.isna()]
        if len(bad) > 0:
            logger.warning("Found %d rows with invalid datetime.", len(bad))
        return dt

    def _load_source(self, path: Path, quake_type: str) -> pd.DataFrame:
        logger.info("Loading %s events from %s", quake_type, path)
        df = pd.read_csv(path)
        df["type"] = quake_type
        if "datetime" in df.columns:
            df["datetime"] = self._safe_datetime(df)
        return df

    def run(self) -> pd.DataFrame:
        cfg = self.config
        df_tect = self._load_source(cfg.tectonic_path, "tectonic")
        df_volc = self._load_source(cfg.volcanic_path, "volcanic")

        events = (
            pd.concat([df_tect, df_volc], ignore_index=True, sort=False)
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        for col in ["latitude", "longitude", "magnitude", "depth_km"]:
            if col in events.columns:
                events[col] = pd.to_numeric(events[col], errors="coerce")

        events = events.dropna(subset=["datetime"])
        if {"latitude", "longitude"} <= set(events.columns):
            events = events.dropna(subset=["latitude", "longitude"], how="any")

        out = cfg.output_path
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            events.to_parquet(out, engine="fastparquet")
            logger.info("Wrote combined events to %s", out)
        except Exception as exc:
            logger.warning("Parquet failed (%s), writing CSV instead.", exc)
            csv_path = out.with_suffix(".csv")
            events.to_csv(csv_path, index=False)
            logger.info("Wrote combined events to %s", csv_path)

        return events
