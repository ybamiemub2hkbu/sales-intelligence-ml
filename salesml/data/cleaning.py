"""Validate and clean the raw export, then derive analysis-ready columns.

Every fix is counted in a data-quality report so analysts can see exactly
what was changed - silent cleaning is how dashboards end up disagreeing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, save_table

log = get_logger(__name__)

REQUIRED_COLUMNS = ["order_id", "line_number", "order_date", "customer_id", "product_id",
                    "category", "channel", "region", "quantity", "list_price",
                    "discount_pct", "unit_price", "unit_cost", "is_returned"]


def load_raw(cfg: Config) -> dict[str, pd.DataFrame]:
    raw = cfg.raw_dir
    if not (raw / "transactions.csv").exists():
        raise FileNotFoundError(f"No raw data in {raw}. Run `python main.py --steps generate` first.")
    return dict(
        transactions=pd.read_csv(raw / "transactions.csv", parse_dates=["order_date"],
                                 dtype={"promo_id": "string"}),
        customers=pd.read_csv(raw / "customers.csv", parse_dates=["signup_date"]),
        products=pd.read_csv(raw / "products.csv"),
        promotions=pd.read_csv(raw / "promotions.csv", parse_dates=["start_date", "end_date"]),
    )


def clean_transactions(tx: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = set(REQUIRED_COLUMNS) - set(tx.columns)
    if missing:
        raise ValueError(f"transactions missing columns: {sorted(missing)}")

    report = []

    def note(check, n, action):
        report.append(dict(check=check, rows_affected=int(n), action=action))

    n0 = len(tx)
    note("rows in raw export", n0, "-")

    # 1. exact duplicate lines (double-loaded batches)
    dup = tx.duplicated()
    note("exact duplicate rows", dup.sum(), "dropped")
    tx = tx[~dup].copy()

    # 2. duplicate keys that are not exact duplicates -> keep first
    key_dup = tx.duplicated(["order_id", "line_number"])
    note("duplicate (order_id, line_number) keys", key_dup.sum(), "kept first occurrence")
    tx = tx[~key_dup]

    # 3. negative quantities: sign-flip entry errors (returns are flagged separately)
    neg = tx["quantity"] <= 0
    note("non-positive quantity", neg.sum(), "absolute value taken (entry error, returns are flagged separately)")
    tx.loc[neg, "quantity"] = tx.loc[neg, "quantity"].abs()

    # 4. unit price keyed in cents: > 20x the list-price-after-discount
    expected = tx["list_price"] * (1 - tx["discount_pct"])
    ratio = tx["unit_price"] / expected
    cents = ratio.between(90, 110)
    note("unit price keyed in cents (~100x expected)", cents.sum(), "divided by 100")
    tx.loc[cents, "unit_price"] = (tx.loc[cents, "unit_price"] / 100).round(2)
    other_bad = ~cents & ((ratio > 1.5) | (ratio < 0.5))
    note("other unit-price outliers", other_bad.sum(), "replaced by list price x (1 - discount)")
    tx.loc[other_bad, "unit_price"] = expected[other_bad].round(2)

    # 5. missing channel -> customer's most frequent channel
    miss = tx["channel"].isna()
    mode = (tx.dropna(subset=["channel"]).groupby("customer_id")["channel"]
            .agg(lambda s: s.value_counts().index[0]))
    tx.loc[miss, "channel"] = tx.loc[miss, "customer_id"].map(mode)
    still = tx["channel"].isna()
    tx.loc[still, "channel"] = "Online"
    note("missing channel", miss.sum(), "imputed with customer's modal channel")

    # 6. referential integrity
    orphan = ~tx["product_id"].isin(products["product_id"])
    note("unknown product_id", orphan.sum(), "dropped")
    tx = tx[~orphan]

    # ---- derived columns (recomputed from clean inputs) ----
    tx["revenue"] = (tx["unit_price"] * tx["quantity"]).round(2)
    tx["cost"] = (tx["unit_cost"] * tx["quantity"]).round(2)
    tx["discount_amount"] = ((tx["list_price"] - tx["unit_price"]) * tx["quantity"]).round(2)
    tx["net_revenue"] = np.where(tx["is_returned"] == 1, 0.0, tx["revenue"])
    tx["gross_profit"] = np.where(tx["is_returned"] == 1, 0.0, tx["revenue"] - tx["cost"])
    tx["promo_id"] = tx["promo_id"].fillna("")
    tx["on_promo"] = (tx["promo_id"] != "").astype(int)
    tx["week_start"] = tx["order_date"] - pd.to_timedelta(tx["order_date"].dt.dayofweek, unit="D")
    tx["month"] = tx["order_date"].dt.to_period("M").dt.to_timestamp()
    tx = tx.sort_values(["order_date", "order_id", "line_number"]).reset_index(drop=True)

    note("rows after cleaning", len(tx), f"{n0 - len(tx)} rows removed in total")
    return tx, pd.DataFrame(report)


def build_processed(cfg: Config) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    raw = load_raw(cfg)
    tx, dq = clean_transactions(raw["transactions"], raw["products"])
    for _, r in dq.iterrows():
        log.info("DQ | %-45s %7d | %s", r.check, r.rows_affected, r.action)
    save_table(dq, cfg.tables_dir / "data_quality_report.csv")
    tx.to_pickle(cfg.processed_dir / "transactions.pkl")
    for name in ("customers", "products", "promotions"):
        raw[name].to_pickle(cfg.processed_dir / f"{name}.pkl")
    return dict(transactions=tx, customers=raw["customers"], products=raw["products"],
                promotions=raw["promotions"], data_quality=dq)


def load_processed(cfg: Config) -> dict[str, pd.DataFrame]:
    p = cfg.processed_dir
    if not (p / "transactions.pkl").exists():
        return build_processed(cfg)
    return {name: pd.read_pickle(p / f"{name}.pkl")
            for name in ("transactions", "customers", "products", "promotions")}
