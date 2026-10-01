import numpy as np
import pandas as pd

from salesml.data.cleaning import REQUIRED_COLUMNS, clean_transactions
from salesml.data.generator import SalesDataGenerator, promotion_calendar, thanksgiving


def test_generator_schema_and_sanity(raw):
    tx = raw["transactions"]
    assert set(REQUIRED_COLUMNS) <= set(tx.columns)
    assert len(tx) > 1000
    assert tx["product_id"].isin(raw["products"]["product_id"]).all()
    assert tx["customer_id"].isin(raw["customers"]["customer_id"]).all()
    assert tx["discount_pct"].between(0, 0.5).all()
    # hidden simulation columns never leak into the customer table
    assert not any(c.startswith("_") for c in raw["customers"].columns)


def test_generator_is_deterministic(small_cfg):
    a = SalesDataGenerator(small_cfg).generate()["transactions"]
    b = SalesDataGenerator(small_cfg).generate()["transactions"]
    pd.testing.assert_frame_equal(a, b)


def test_dirty_records_are_injected(raw):
    tx = raw["transactions"]
    assert tx.duplicated().sum() > 0
    assert (tx["quantity"] < 0).sum() > 0
    assert tx["channel"].isna().sum() > 0


def test_cleaning_fixes_every_injected_issue(clean_tx):
    tx = clean_tx
    assert not tx.duplicated(["order_id", "line_number"]).any()
    assert (tx["quantity"] > 0).all()
    assert tx["channel"].notna().all()
    ratio = tx["unit_price"] / (tx["list_price"] * (1 - tx["discount_pct"]))
    assert ratio.between(0.95, 1.05).all()
    np.testing.assert_allclose(tx["revenue"], (tx["unit_price"] * tx["quantity"]).round(2))
    assert (tx.loc[tx["is_returned"] == 1, "net_revenue"] == 0).all()


def test_cleaning_rejects_missing_columns(raw):
    bad = raw["transactions"].drop(columns=["unit_cost"])
    try:
        clean_transactions(bad, raw["products"])
    except ValueError as e:
        assert "unit_cost" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_calendar_helpers():
    assert thanksgiving(2024) == pd.Timestamp("2024-11-28")
    assert thanksgiving(2025) == pd.Timestamp("2025-11-27")
    promos = promotion_calendar([2025])
    bf = promos[promos["promo_name"] == "Black Friday Week"].iloc[0]
    assert bf["start_date"] <= thanksgiving(2025) <= bf["end_date"]
