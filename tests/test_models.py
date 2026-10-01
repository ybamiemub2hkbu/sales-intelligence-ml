import numpy as np
import pandas as pd
import pytest

from salesml.models.anomaly import robust_z, to_incidents
from salesml.models.basket import pair_rules
from salesml.models.churn import retention_profit_curve
from salesml.models.elasticity import _within_estimate
from salesml.models.forecasting import repair_incidents, seasonal_naive
from salesml.utils import bias, top_decile_capture, wape


def test_metrics():
    assert wape([10, 10], [12, 8]) == pytest.approx(0.2)
    assert bias([10, 10], [12, 12]) == pytest.approx(0.2)
    y = np.r_[np.zeros(90), np.full(10, 10.0)]
    assert top_decile_capture(y, y) == pytest.approx(1.0)


def test_pair_rules_known_lift():
    # 100 orders: A in 20, B in 20, A&B together in 10 -> lift = (10/20) / (20/100) = 2.5
    rows = []
    for i in range(100):
        if i < 10:
            rows += [(i, "A"), (i, "B")]
        elif i < 20:
            rows += [(i, "A"), (i, "C")]
        elif i < 30:
            rows += [(i, "B"), (i, "C")]
        else:
            rows += [(i, "D")]
    df = pd.DataFrame(rows, columns=["order_id", "item"])
    r = pair_rules(df, "item", min_pair_orders=1).set_index(["antecedent", "consequent"])
    assert r.loc[("A", "B"), "lift"] == pytest.approx(2.5)
    assert r.loc[("A", "B"), "confidence"] == pytest.approx(0.5)
    assert r.loc[("A", "B"), "support"] == pytest.approx(0.10)


def test_fixed_effects_elasticity_recovers_truth():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(12):
        pop = rng.normal(3, 1)
        for w in range(80):
            shock = np.sin(w / 5)                      # common weekly traffic shock
            logp = np.log(20 + 5 * p) + rng.normal(0, 0.1)
            rows.append(dict(product_id=p, week_start=w, log_p=logp,
                             log_q=pop - 1.7 * logp + shock + rng.normal(0, 0.05)))
    est = _within_estimate(pd.DataFrame(rows))
    assert est == pytest.approx(-1.7, abs=0.05)


def test_retention_curve_prefers_targeting():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    churned = (rng.random(2000) < p).astype(int)
    value = np.full(2000, 50.0)
    share, net, _ = retention_profit_curve(p, churned, value, cost=8, success=0.25)
    assert net.max() > net[-1]                  # stopping early beats contacting everyone
    assert 0 < share[np.argmax(net)] < 1


def test_robust_z_and_incident_merge():
    r = np.r_[np.random.default_rng(2).normal(0, 1, 200), [12.0]]
    assert robust_z(r)[-1] > 8
    days = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-02-01"])
    flags = pd.DataFrame(dict(series="Total", direction="drop", date=days, actual=[1, 1, 1, 1],
                              expected=[5, 5, 5, 5], z=[-4, -5, -4, -6]))
    inc = to_incidents(flags)
    assert len(inc) == 2
    assert inc.loc[inc["days"] == 3, "impact"].iloc[0] == -12


def test_forecast_helpers():
    weeks = pd.date_range("2023-01-02", periods=120, freq="7D")
    base = 100 + 20 * np.sin(np.arange(120) * 2 * np.pi / 52)
    s = pd.DataFrame({"X": base}, index=weeks)
    sn = seasonal_naive(s, 110, 5)
    assert len(sn) == 5 and (sn["forecast"] > 0).all()
    # a one-off 90% dip with no promotion should be repaired
    dipped = s.copy()
    dipped.iloc[80, 0] *= 0.1
    cal = dict(disc=pd.DataFrame(0.0, index=weeks, columns=["X"]), sitewide=pd.Series(0.0, index=weeks))
    fixed, flags = repair_incidents(dipped, cal)
    assert weeks[80] in set(flags["week_start"])
    assert fixed.iloc[80, 0] > 50
