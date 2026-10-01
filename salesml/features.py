"""Point-in-time customer features shared by churn, CLV and segmentation.

``customer_snapshot(tx, as_of)`` only ever looks at transactions strictly
*before* ``as_of``. That single rule is what prevents target leakage when we
train on past snapshots and score the present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days", "frequency", "monetary", "avg_order_value", "tenure_days",
    "orders_90d", "orders_prev_90d", "revenue_90d", "revenue_365d", "order_trend",
    "avg_interpurchase_days", "recency_vs_cadence", "discount_share", "avg_discount",
    "n_categories", "digital_share", "mobile_share", "return_rate", "items_per_order",
    "promo_order_share", "days_since_first_promo_only", "loyalty_member",
]
CATEGORICAL_FEATURES = ["region", "age_band"]


def customer_snapshot(tx: pd.DataFrame, as_of: pd.Timestamp,
                      customers: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per customer with >= 1 order before ``as_of``."""
    as_of = pd.Timestamp(as_of)
    h = tx[tx["order_date"] < as_of]
    if h.empty:
        raise ValueError(f"no transactions before {as_of.date()}")

    orders = (h.groupby(["customer_id", "order_id"], sort=False)
              .agg(date=("order_date", "first"), net=("net_revenue", "sum"),
                   gross=("revenue", "sum"), items=("quantity", "sum"),
                   disc_amt=("discount_amount", "sum"), promo=("on_promo", "max"),
                   channel=("channel", "first"))
              .reset_index())
    g = orders.groupby("customer_id")
    f = pd.DataFrame({
        "first_date": g["date"].min(), "last_date": g["date"].max(),
        "frequency": g.size(), "monetary": g["net"].sum(), "gross": g["gross"].sum(),
        "items": g["items"].sum(), "disc_amt": g["disc_amt"].sum(),
        "promo_orders": g["promo"].sum(),
    })
    f["recency_days"] = (as_of - f["last_date"]).dt.days
    f["tenure_days"] = (as_of - f["first_date"]).dt.days
    f["avg_order_value"] = f["monetary"] / f["frequency"]
    f["items_per_order"] = f["items"] / f["frequency"]
    f["promo_order_share"] = f["promo_orders"] / f["frequency"]

    # inter-purchase cadence; single-order customers get tenure as cadence
    span = (f["last_date"] - f["first_date"]).dt.days
    f["avg_interpurchase_days"] = np.where(f["frequency"] > 1, span / (f["frequency"] - 1).clip(lower=1),
                                           f["tenure_days"].clip(lower=30))
    f["recency_vs_cadence"] = f["recency_days"] / f["avg_interpurchase_days"].clip(lower=7)

    def window(lo, hi):
        m = (orders["date"] >= as_of - pd.Timedelta(days=lo)) & (orders["date"] < as_of - pd.Timedelta(days=hi))
        return orders[m].groupby("customer_id")

    w90, wprev, w365 = window(90, 0), window(180, 90), window(365, 0)
    f["orders_90d"] = w90.size().reindex(f.index, fill_value=0)
    f["orders_prev_90d"] = wprev.size().reindex(f.index, fill_value=0)
    f["revenue_90d"] = w90["net"].sum().reindex(f.index, fill_value=0)
    f["revenue_365d"] = w365["net"].sum().reindex(f.index, fill_value=0)
    f["order_trend"] = (f["orders_90d"] + 1) / (f["orders_prev_90d"] + 1)

    lines = h.groupby("customer_id")
    f["discount_share"] = (h["discount_pct"].gt(0) * h["revenue"]).groupby(h["customer_id"]).sum() / lines["revenue"].sum()
    f["avg_discount"] = f["disc_amt"] / (f["gross"] + f["disc_amt"]).clip(lower=1e-9)
    f["n_categories"] = lines["category"].nunique()
    f["return_rate"] = lines["is_returned"].mean()
    f["digital_share"] = orders["channel"].isin(["Online", "Mobile App"]).groupby(orders["customer_id"]).mean()
    f["mobile_share"] = orders["channel"].eq("Mobile App").groupby(orders["customer_id"]).mean()
    # customers who only ever bought on promotion
    f["days_since_first_promo_only"] = np.where(f["promo_order_share"] >= 0.999, f["tenure_days"], 0)

    if customers is not None:
        c = customers.set_index("customer_id")
        f = f.join(c[["region", "age_band", "loyalty_member"]], how="left")
        f["loyalty_member"] = f["loyalty_member"].astype(float)
    else:
        f["region"], f["age_band"], f["loyalty_member"] = "NA", "NA", 0.0

    f["as_of"] = as_of
    keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["as_of", "first_date", "last_date"]
    return f[keep].fillna(0)


def future_outcomes(tx: pd.DataFrame, as_of: pd.Timestamp, horizon_days: int) -> pd.DataFrame:
    """Orders and net revenue per customer in [as_of, as_of + horizon)."""
    as_of = pd.Timestamp(as_of)
    m = (tx["order_date"] >= as_of) & (tx["order_date"] < as_of + pd.Timedelta(days=horizon_days))
    fut = tx[m]
    return pd.DataFrame({
        "future_orders": fut.groupby("customer_id")["order_id"].nunique(),
        "future_revenue": fut.groupby("customer_id")["net_revenue"].sum(),
    })


def model_matrix(feat: pd.DataFrame) -> pd.DataFrame:
    """Numeric design matrix with one-hot categoricals (stable column set)."""
    X = feat[NUMERIC_FEATURES].astype(float).copy()
    for col, levels in (("region", ["North", "South", "East", "West"]),
                        ("age_band", ["18-24", "25-34", "35-44", "45-54", "55+"])):
        for lv in levels:
            X[f"{col}={lv}"] = (feat[col] == lv).astype(float)
    return X
