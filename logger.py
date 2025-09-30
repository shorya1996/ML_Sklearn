# util/logger.py
from logging import Logger, getLogger, StreamHandler, Formatter, FileHandler, INFO
from pathlib import Path
from typing import Optional

def setup_logger(name: str = "fraud_app", level: int = INFO, log_file: Optional[str] = None) -> Logger:
    logger = getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = Formatter("%(asctime)s | %(levelname)7s | %(name)s | %(message)s")
    ch = StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger
