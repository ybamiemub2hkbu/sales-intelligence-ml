import pandas as pd

from salesml.features import NUMERIC_FEATURES, customer_snapshot, future_outcomes, model_matrix


def test_snapshot_never_uses_future_data(clean_tx, raw):
    as_of = pd.Timestamp("2024-01-01")
    base = customer_snapshot(clean_tx, as_of, raw["customers"])
    # wildly alter everything on/after as_of: features must not change
    tampered = clean_tx.copy()
    future = tampered["order_date"] >= as_of
    tampered.loc[future, ["net_revenue", "revenue", "quantity"]] *= 100
    after = customer_snapshot(tampered, as_of, raw["customers"])
    pd.testing.assert_frame_equal(base, after)


def test_snapshot_values_on_toy_data():
    tx = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B"],
        "order_id": ["o1", "o2", "o3", "o4"],
        "order_date": pd.to_datetime(["2024-01-01", "2024-01-11", "2024-03-01", "2024-02-20"]),
        "net_revenue": [100.0, 50.0, 70.0, 30.0], "revenue": [100.0, 50.0, 70.0, 30.0],
        "quantity": [1, 1, 1, 1], "discount_amount": [0.0, 10.0, 0.0, 0.0], "discount_pct": [0, 0.2, 0, 0],
        "on_promo": [0, 1, 0, 0], "channel": ["Online", "In-Store", "Online", "Mobile App"],
        "category": ["Apparel", "Apparel", "Electronics", "Apparel"], "is_returned": [0, 0, 0, 0],
    })
    f = customer_snapshot(tx, pd.Timestamp("2024-03-01"))   # o3 is ON as_of -> excluded
    a = f.loc["A"]
    assert a["frequency"] == 2
    assert a["monetary"] == 150
    assert a["recency_days"] == (pd.Timestamp("2024-03-01") - pd.Timestamp("2024-01-11")).days
    assert a["avg_interpurchase_days"] == 10
    assert a["n_categories"] == 1
    assert a["digital_share"] == 0.5
    out = future_outcomes(tx, pd.Timestamp("2024-03-01"), 30)
    assert out.loc["A", "future_orders"] == 1 and "B" not in out.index


def test_model_matrix_has_stable_columns(clean_tx, raw):
    X1 = model_matrix(customer_snapshot(clean_tx, pd.Timestamp("2023-09-01"), raw["customers"]))
    X2 = model_matrix(customer_snapshot(clean_tx, pd.Timestamp("2024-06-01"), raw["customers"]))
    assert list(X1.columns) == list(X2.columns)
    assert set(NUMERIC_FEATURES) <= set(X1.columns)
    assert X1.notna().all().all()
