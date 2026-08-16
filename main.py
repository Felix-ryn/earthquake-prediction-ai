from src.config import build_pipeline_from_env
from src.pipeline import EarthquakePipeline

pipe_cfg = build_pipeline_from_env()
pipeline = EarthquakePipeline(pipe_cfg)

result = pipeline.run_all()
