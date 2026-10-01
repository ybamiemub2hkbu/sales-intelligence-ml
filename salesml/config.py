"""Central configuration for the Sales Intelligence pipeline.

Every tunable number in the project lives here so that experiments are
reproducible and a single place documents the business assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    # ---- reproducibility -------------------------------------------------
    seed: int = 42

    # ---- simulated company ------------------------------------------------
    company_name: str = "Meridian Home & Outdoor"
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    n_customers: int = 18_000

    # ---- paths ------------------------------------------------------------
    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"
    models_dir: Path = ROOT / "models"

    # ---- modelling horizons ----------------------------------------------
    churn_horizon_days: int = 120          # "churned" = no order in next N days
    churn_snapshot_step_days: int = 60     # spacing of stacked training snapshots
    churn_n_train_snapshots: int = 3
    clv_horizon_days: int = 180            # predict net revenue over next N days
    forecast_horizon_weeks: int = 13       # one business quarter
    forecast_backtest_origins: int = 4     # rolling-origin backtests (one year)

    # ---- business assumptions used to turn model output into decisions ----
    retention_offer_cost: float = 8.0      # $ cost per customer contacted
    retention_success_rate: float = 0.25   # share of true churners saved by offer
    price_change_guardrail: float = 0.10   # max +/- list-price move we recommend
    basket_min_pair_orders: int = 25       # min co-occurrences for a rule

    # ---- segmentation ------------------------------------------------------
    segment_k_range: tuple = (4, 6)

    # ---- anomaly detection ------------------------------------------------
    anomaly_z_threshold: float = 3.5

    fast: bool = False                      # smaller bootstrap/model sizes

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.reports_dir / "tables"

    def ensure_dirs(self) -> None:
        for p in (self.raw_dir, self.processed_dir, self.figures_dir,
                  self.tables_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)
