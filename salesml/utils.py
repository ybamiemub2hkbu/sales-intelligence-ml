"""Shared helpers: logging, metrics, persistence."""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger(name)


@contextmanager
def timer(logger: logging.Logger, label: str):
    t0 = time.perf_counter()
    logger.info("%s ...", label)
    yield
    logger.info("%s done in %.1fs", label, time.perf_counter() - t0)


# ---------------------------------------------------------------- metrics --
def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error - robust for retail volumes."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.abs(y_true - y_pred).sum() / max(np.abs(y_true).sum(), 1e-9))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true, y_pred) -> float:
    """Positive = over-forecast."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float((y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9))


def top_decile_capture(y_true, y_score) -> float:
    """Share of total actual value held by the top 10% ranked by score."""
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    n = max(int(len(y_true) * 0.1), 1)
    idx = np.argsort(-y_score)[:n]
    return float(y_true[idx].sum() / max(y_true.sum(), 1e-9))


# ------------------------------------------------------------ persistence --
def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return o.strftime("%Y-%m-%d")
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    path.write_text(json.dumps(obj, indent=2, default=_default))
    return path


def load_json(path: Path):
    return json.loads(Path(path).read_text())


def money(x: float, decimals: int = 0) -> str:
    """Human friendly currency: $1.23M, $45.6K, $789."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e6:
        return f"{sign}${x / 1e6:.2f}M"
    if x >= 1e4:
        return f"{sign}${x / 1e3:.1f}K"
    return f"{sign}${x:,.{decimals}f}"


def pct(x: float, decimals: int = 1) -> str:
    return f"{x * 100:.{decimals}f}%"
