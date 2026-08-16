import logging
from typing import Optional
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name: Optional[str] = "app", level: str = "INFO") -> logging.Logger:
    """
    Setup dan kembalikan logger dengan konfigurasi standar.
    
    Args:
        name: Nama logger.
        level: Level logging ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    Returns:
        logging.Logger: Objek logger yang siap digunakan.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_value = getattr(logging, str(level).upper(), logging.INFO)
        logger.setLevel(level_value)

        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        ch.setLevel(level_value)
        logger.addHandler(ch)

        log_dir = os.path.join(os.getcwd(), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
        fh.setFormatter(formatter)
        fh.setLevel(level_value)
        logger.addHandler(fh)

        logger.propagate = False
        
    return logger