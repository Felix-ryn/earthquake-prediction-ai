from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_env(key: str, default: Any) -> Any:
	val = os.getenv(key)
	if val is None:
		return default
	if isinstance(default, list):
		try:
			if val.strip().startswith("["):
				return json.loads(val)
			return [v.strip() for v in val.split(",") if v.strip()]
		except Exception:
			return default
	if isinstance(default, bool):
		return val.lower() in ("1", "true", "yes", "on")
	try:
		if isinstance(default, int):
			return int(val)
		if isinstance(default, float):
			return float(val)
	except Exception:
		return default
	return val


def _load_env_file(path: Optional[Path] = None) -> None:
    """
    Minimal .env loader without external deps. Parses KEY=VALUE lines.
    Does not override already-set environment variables.
    """
    if path is None:
        path = Path.cwd() / ".env"
    try:
        if not Path(path).exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except Exception:
        pass


@dataclass	
class IngestConfig:
    tectonic_path: Path
    volcanic_path: Path
    output_path: Path 


@dataclass
class ACOConfig:
    n_ants: int = 40
    n_iters: int = 25
    neighbor_k: int = 10
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.3
    walk_length: int = 15
    seed: int = 42
    output_path: Path | None = None 
    geojson_path: Path | None = None
    
@dataclass
class GAConfig:
	freq: str = "M"   
	min_events: int = 5       
	pop_size: int = 40
	n_generations: int = 120
	elite_frac: float = 0.2
	mutation_std_deg: float = 20.0
	seed: int = 42    	
	output_path: Path | None = None 
	geojson_path: Path | None = None    

@dataclass    
class TrainConfig:
    events_path: Path
    model_dir: Path = Path("models")
    seq_len: int = 10
    batch_size: int = 32
    epochs_cnn: int = 25
    epochs_lstm: int = 25
    learning_rate: float = 1e-3
    test_size: float = 0.3
    random_state: int = 42
    lambda_reg: float = 0.5
    lambda_cls: float = 0.5
    
@dataclass
class EvalConfig:
    events_path: Path
    metrics_path: Path = Path("outputs/metrics_nb.json")
    metrics_models_path: Path = Path("outputs/metrics_models.json")
    metrics_web_path: Path = Path("web/data/metrics_models.json")
    model_dir: Path = Path("models")
    seq_len: int = 10
    test_size: float = 0.3
    random_state: int = 42
    stratify: bool = True
    

@dataclass
class PredictConfig:
    events_path: Path
    model_dir: Path
    seq_len: int = 10
    roi_path: Path = Path("outputs/inference_roi.csv")
    roi_geojson_path: Path = Path("outputs/inference_roi.geojson")

@dataclass
class IOConfig:
    tectonic_csv: Path
    volcanic_csv: Path
    events_path: Path = Path("outputs/events.parquet")
    events_aco_path: Path = Path("outputs/events_aco.parquet")
    events_aco_ga_path: Path = Path("outputs/events_aco_ga.parquet")
    events_aco_geojson_path: Path = Path("web/data/events_aco.geojson")
    ga_vectors_geojson_path: Path = Path("web/data/ga_vectors.geojson")
    models_dir: Path = Path("models")
    nb_metrics_path: Path = Path("outputs/metrics_nb.json")
    metrics_models_path: Path = Path("outputs/metrics_models.json")
    metrics_models_web_path: Path = Path("web/data/metrics_models.json")
    roi_path: Path = Path("outputs/inference_roi.csv")
    roi_geojson_path: Path = Path("web/data/inference_roi.geojson")


@dataclass
class PipelineConfig:
    io: IOConfig
    aco: ACOConfig = field(default_factory=ACOConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    train: Optional[TrainConfig] = None
    eval: Optional[EvalConfig] = None
    predict: Optional[PredictConfig] = None
    seq_len: int = 10

    def build_train_config(self) -> TrainConfig:
        if self.train is not None:
            return self.train
        return TrainConfig(
            events_path=self.io.events_aco_ga_path,
            model_dir=self.io.models_dir,
            seq_len=self.seq_len,
        )

    def build_eval_config(self) -> EvalConfig:
        if self.eval is not None:
            return self.eval
        return EvalConfig(
            events_path=self.io.events_aco_ga_path,
            metrics_path=self.io.nb_metrics_path,
            metrics_models_path=self.io.metrics_models_path,
            metrics_web_path=self.io.metrics_models_web_path,
            model_dir=self.io.models_dir,
            seq_len=self.seq_len,
        )

    def build_predict_config(self) -> PredictConfig:
        if self.predict is not None:
            return self.predict
        return PredictConfig(
            events_path=self.io.events_aco_ga_path,
            model_dir=self.io.models_dir,
            seq_len=self.seq_len,
            roi_path=self.io.roi_path,
            roi_geojson_path=self.io.roi_geojson_path,
        )


# ----------- Env-driven builders -----------

def build_io_from_env() -> "IOConfig":
    _load_env_file()
    return IOConfig(
        tectonic_csv=Path(_get_env("IO_TECTONIC_CSV", "data/tectonic_events.csv")),
        volcanic_csv=Path(_get_env("IO_VOLCANIC_CSV", "data/volcanic_events.csv")),
        events_path=Path(_get_env("IO_EVENTS_PATH", "outputs/events.parquet")),
        events_aco_path=Path(_get_env("IO_EVENTS_ACO_PATH", "outputs/events_aco.parquet")),
        events_aco_ga_path=Path(_get_env("IO_EVENTS_ACO_GA_PATH", "outputs/events_aco_ga.parquet")),
        events_aco_geojson_path=Path(_get_env("IO_EVENTS_ACO_GEOJSON_PATH", "web/data/events_aco.geojson")),
        ga_vectors_geojson_path=Path(_get_env("IO_GA_VECTORS_GEOJSON_PATH", "web/data/ga_vectors.geojson")),
        models_dir=Path(_get_env("IO_MODELS_DIR", "models")),
        nb_metrics_path=Path(_get_env("IO_NB_METRICS_PATH", "outputs/metrics_nb.json")),
        metrics_models_path=Path(_get_env("IO_METRICS_MODELS_PATH", "outputs/metrics_models.json")),
        metrics_models_web_path=Path(_get_env("IO_METRICS_MODELS_WEB_PATH", "web/data/metrics_models.json")),
        roi_path=Path(_get_env("IO_ROI_PATH", "outputs/inference_roi.csv")),
        roi_geojson_path=Path(_get_env("IO_ROI_GEOJSON_PATH", "web/data/inference_roi.geojson")),
    )


def build_aco_from_env() -> "ACOConfig":
    _load_env_file()
    return ACOConfig(
        n_ants=_get_env("ACO_N_ANTS", 40),
        n_iters=_get_env("ACO_N_ITERS", 25),
        neighbor_k=_get_env("ACO_NEIGHBOR_K", 10),
        alpha=_get_env("ACO_ALPHA", 1.0),
        beta=_get_env("ACO_BETA", 2.0),
        rho=_get_env("ACO_RHO", 0.3),
        walk_length=_get_env("ACO_WALK_LENGTH", 15),
        seed=_get_env("ACO_SEED", 42),
        output_path=None,
        geojson_path=None,
    )


def build_ga_from_env() -> "GAConfig":
    _load_env_file()
    return GAConfig(
        freq=_get_env("GA_FREQ", "M"),
        min_events=_get_env("GA_MIN_EVENTS", 5),
        pop_size=_get_env("GA_POP_SIZE", 40),
        n_generations=_get_env("GA_N_GENERATIONS", 120),
        elite_frac=_get_env("GA_ELITE_FRAC", 0.2),
        mutation_std_deg=_get_env("GA_MUTATION_STD_DEG", 20.0),
        seed=_get_env("GA_SEED", 42),
        output_path=None,
        geojson_path=None,
    )


def build_train_from_env(io: Optional["IOConfig"] = None) -> Optional["TrainConfig"]:
    _load_env_file()
    if _get_env("TRAIN_ENABLED", "1") in ("0", 0, False):
        return None
    events_path = Path(_get_env("TRAIN_EVENTS_PATH", str(io.events_aco_ga_path if io else "outputs/events_aco_ga.parquet")))
    return TrainConfig(
        events_path=events_path,
        model_dir=Path(_get_env("TRAIN_MODEL_DIR", str((io.models_dir if io else Path("models"))))),
        seq_len=_get_env("TRAIN_SEQ_LEN", 10),
        batch_size=_get_env("TRAIN_BATCH_SIZE", 32),
        epochs_cnn=_get_env("TRAIN_EPOCHS_CNN", 25),
        epochs_lstm=_get_env("TRAIN_EPOCHS_LSTM", 25),
        learning_rate=_get_env("TRAIN_LR", 1e-3),
        test_size=_get_env("TRAIN_TEST_SIZE", 0.3),
        random_state=_get_env("TRAIN_RANDOM_STATE", 42),
        lambda_reg=_get_env("TRAIN_LAMBDA_REG", 0.5),
        lambda_cls=_get_env("TRAIN_LAMBDA_CLS", 0.5),
    )


def build_eval_from_env(io: Optional["IOConfig"] = None) -> Optional["EvalConfig"]:
    _load_env_file()
    if _get_env("EVAL_ENABLED", "1") in ("0", 0, False):
        return None
    return EvalConfig(
        events_path=Path(_get_env("EVAL_EVENTS_PATH", str(io.events_aco_ga_path if io else "outputs/events_aco_ga.parquet"))),
        metrics_path=Path(_get_env("EVAL_METRICS_PATH", str(io.nb_metrics_path if io else "outputs/metrics_nb.json"))),
        metrics_models_path=Path(_get_env("EVAL_METRICS_MODELS_PATH", str(io.metrics_models_path if io else "outputs/metrics_models.json"))),
        metrics_web_path=Path(_get_env("EVAL_METRICS_WEB_PATH", str(io.metrics_models_web_path if io else "web/data/metrics_models.json"))),
        model_dir=Path(_get_env("EVAL_MODEL_DIR", str(io.models_dir if io else "models"))),
        seq_len=_get_env("EVAL_SEQ_LEN", 10),
        test_size=_get_env("EVAL_TEST_SIZE", 0.3),
        random_state=_get_env("EVAL_RANDOM_STATE", 42),
        stratify=_get_env("EVAL_STRATIFY", True),
    )


def build_predict_from_env(io: Optional["IOConfig"] = None, default_seq_len: int = 10) -> Optional["PredictConfig"]:
    _load_env_file()
    if _get_env("PREDICT_ENABLED", "1") in ("0", 0, False):
        return None
    return PredictConfig(
        events_path=Path(_get_env("PREDICT_EVENTS_PATH", str(io.events_aco_ga_path if io else "outputs/events_aco_ga.parquet"))),
        model_dir=Path(_get_env("PREDICT_MODEL_DIR", str(io.models_dir if io else "models"))),
        seq_len=_get_env("PREDICT_SEQ_LEN", default_seq_len),
        roi_path=Path(_get_env("PREDICT_ROI_PATH", str(io.roi_path if io else "outputs/inference_roi.csv"))),
        roi_geojson_path=Path(_get_env("PREDICT_ROI_GEOJSON_PATH", str(io.roi_geojson_path if io else "web/data/inference_roi.geojson"))),
    )


def build_pipeline_from_env() -> "PipelineConfig":
    _load_env_file()
    io = build_io_from_env()
    seq_len = _get_env("PIPELINE_SEQ_LEN", 10)
    return PipelineConfig(
        io=io,
        aco=build_aco_from_env(),
        ga=build_ga_from_env(),
        train=build_train_from_env(io),
        eval=build_eval_from_env(io),
        predict=build_predict_from_env(io, default_seq_len=seq_len),
        seq_len=seq_len,
    )