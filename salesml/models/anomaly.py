"""Daily sales anomaly detection and incident log.

For every monitored series (total, each channel, category and region) a
gradient-boosted model learns the *expected* daily revenue from calendar,
promotion and trend features using out-of-fold predictions, so a day is never
judged by a model that has seen it.  Residuals of rolling 3-day sums are
scored with a robust z-score (median / MAD), consecutive flagged days are merged into incidents,
and each incident gets an estimated revenue impact.

A second, multivariate detector (Isolation Forest over the residual vector of
all series) catches days whose *pattern* across channels and categories is
unusual even when the total looks normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest

from .. import viz
from ..config import Config
from ..data.generator import thanksgiving
from ..utils import get_logger, load_json, save_json, save_table

log = get_logger(__name__)
WINDOW = 3


def known_holidays(days: pd.DatetimeIndex) -> np.ndarray:
    """Dates every retailer expects to be abnormal (holidays and the pre-Christmas
    rush) - never reported as incidents."""
    tg = {y: thanksgiving(y) for y in set(days.year)}
    return np.array([(d.month == 12 and (8 <= d.day <= 25 or d.day == 31)) or (d.month == 1 and d.day == 1)
                     or d == tg[d.year] for d in days])


def daily_features(days: pd.DatetimeIndex, promos: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    f = pd.DataFrame(index=days)
    f["dow"] = days.dayofweek
    f["doy_sin"] = np.sin(2 * np.pi * days.dayofyear / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * days.dayofyear / 365.25)
    f["month"] = days.month
    f["trend"] = np.arange(len(days))
    f["dom"] = days.day
    tg = {y: thanksgiving(y) for y in set(days.year)}
    f["days_to_black_friday"] = [(d - (tg[d.year] + pd.Timedelta(days=1))).days for d in days]
    f["days_to_christmas"] = [(d - pd.Timestamp(year=d.year, month=12, day=25)).days for d in days]
    f["known_holiday"] = known_holidays(days).astype(float)
    f["pre_christmas"] = ((days.month == 12) & (days.day >= 8) & (days.day <= 23)).astype(float)
    f["sitewide_promo"] = 0.0
    f["post_sitewide"] = 0.0
    for c in categories:
        f[f"disc_{c}"] = 0.0
    for p in promos.itertuples(index=False):
        m = (days >= p.start_date) & (days <= p.end_date)
        if p.categories == "ALL":
            f.loc[m, "sitewide_promo"] = p.discount_pct
            after = (days > p.end_date) & (days <= p.end_date + pd.Timedelta(days=10))
            f.loc[after, "post_sitewide"] = 1.0
            cats = categories
        else:
            cats = [c for c in p.categories.split("|") if c in categories]
        for c in cats:
            f.loc[m, f"disc_{c}"] = np.maximum(f.loc[m, f"disc_{c}"], p.discount_pct)
    return f


def expected_oof(y: pd.Series, X: pd.DataFrame, seed: int, fast: bool,
                 exclude: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold expected log revenue with whole calendar months held out.

    Shuffled K-fold would leak: a 3-week stock-out would be mostly in the
    training folds and the model would learn it as normal.  Holding out
    entire months (month_index mod 5, so the same calendar month falls in
    different folds in different years) keeps every incident unseen.
    ``exclude`` drops days already judged anomalous from *training* so one
    incident cannot distort the baseline of another year.
    """
    ly = np.log1p(y.to_numpy())
    idx = y.index
    fold = ((idx.year - idx.year[0]) * 12 + idx.month - 1) % 5
    keep = np.ones(len(y), bool) if exclude is None else ~exclude
    pred = np.zeros(len(y))
    for k in range(5):
        tr, te = np.where((fold != k) & keep)[0], np.where(fold == k)[0]
        m = HistGradientBoostingRegressor(max_iter=100 if fast else 200, learning_rate=0.07,
                                          max_leaf_nodes=15, min_samples_leaf=15, random_state=seed)
        m.fit(X.iloc[tr], ly[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def window_z(s: pd.Series, pred: np.ndarray) -> np.ndarray:
    """Robust z-score of rolling 3-day actual vs expected (log ratio)."""
    act3 = s.rolling(WINDOW, center=True, min_periods=1).sum().to_numpy()
    exp3 = pd.Series(np.expm1(pred), index=s.index).rolling(WINDOW, center=True, min_periods=1).sum().to_numpy()
    k = 0.25 * np.median(exp3)  # pseudo-count so near-empty series cannot explode
    return robust_z(np.log(act3 + k) - np.log(exp3 + k))


def robust_z(r: np.ndarray) -> np.ndarray:
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826
    return (r - med) / max(mad, 1e-9)


def to_incidents(flags: pd.DataFrame) -> pd.DataFrame:
    """Merge consecutive flagged days of the same series & direction."""
    rows = []
    for (series, direction), g in flags.sort_values("date").groupby(["series", "direction"]):
        g = g.sort_values("date")
        block = (g["date"].diff().dt.days.fillna(1) > 2).cumsum()
        for _, b in g.groupby(block):
            rows.append(dict(series=series, direction=direction, start=b["date"].min(), end=b["date"].max(),
                             days=len(b), actual=b["actual"].sum(), expected=b["expected"].sum(),
                             impact=b["actual"].sum() - b["expected"].sum(), peak_z=b["z"].abs().max()))
    inc = pd.DataFrame(rows)
    return inc.sort_values("impact", key=lambda s: s.abs(), ascending=False).reset_index(drop=True) if len(inc) else inc


def run(data: dict, cfg: Config) -> dict:
    tx, promos = data["transactions"], data["promotions"]
    days = pd.date_range(tx["order_date"].min(), tx["order_date"].max(), freq="D")
    categories = viz.CATEGORIES
    X = daily_features(days, promos, categories)

    series = {"Total": tx.groupby("order_date")["revenue"].sum()}
    for dim in ("channel", "category", "region"):
        for k, g in tx.groupby(dim):
            series[f"{dim.title()}: {k}"] = g.groupby("order_date")["revenue"].sum()
    for (r, c), g in tx.groupby(["region", "channel"]):
        series[f"Store: {r} / {c}"] = g.groupby("order_date")["revenue"].sum()
    names = data["products"].set_index("product_id")["product_name"]
    for pid in tx.groupby("product_id")["revenue"].sum().nlargest(15).index:
        g = tx[tx["product_id"] == pid]
        series[f"Product: {names[pid]}"] = g.groupby("order_date")["revenue"].sum()
    holiday = known_holidays(days)
    avg_daily = float(series["Total"].mean())

    flags, resid = [], {}
    expected = {}
    for name, s in series.items():
        s = s.reindex(days, fill_value=0.0)
        # two passes: the second refits without days the first pass found anomalous
        pred = expected_oof(s, X, cfg.seed, cfg.fast)
        z = window_z(s, pred)
        pred = expected_oof(s, X, cfg.seed, cfg.fast, exclude=np.abs(z) > 3.0)
        z = window_z(s, pred)
        resid[name] = z
        expected[name] = np.expm1(pred)
        hit = (np.abs(z) > cfg.anomaly_z_threshold) & ~holiday
        for i in np.where(hit)[0]:
            flags.append(dict(series=name, date=days[i], actual=s.iloc[i], expected=float(np.expm1(pred[i])),
                              z=float(z[i]), direction="spike" if z[i] > 0 else "drop"))
    flags = pd.DataFrame(flags)
    incidents = to_incidents(flags) if len(flags) else pd.DataFrame()
    # materiality: an incident must move at least 15% of an average day's total revenue
    min_impact = 0.15 * avg_daily
    all_incidents = incidents
    incidents = incidents[incidents["impact"].abs() >= min_impact].reset_index(drop=True)

    # multivariate pattern detector
    R = pd.DataFrame(resid, index=days)
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=cfg.seed).fit(R)
    iso_days = R.index[iso.predict(R) == -1]

    # validation against injected incidents
    validation = []
    gt_path = cfg.raw_dir / "ground_truth.json"
    if gt_path.exists() and len(incidents):
        for inc in load_json(gt_path)["incidents"]:
            s, e = pd.Timestamp(inc["start"]), pd.Timestamp(inc["end"])
            overlap = incidents[(incidents["start"] <= e) & (incidents["end"] >= s)]
            in_win = (days >= s) & (days <= e)
            peak = max(resid, key=lambda n: np.abs(resid[n][in_win]).max())
            validation.append(dict(incident=inc["kind"], window=f"{inc['start']} to {inc['end']}",
                                   detected=bool(len(overlap)),
                                   strongest_series=peak,
                                   strongest_abs_z=float(np.abs(resid[peak][in_win]).max()),
                                   series_flagged=", ".join(sorted(overlap["series"].unique()))[:120],
                                   isolation_forest_flag=bool(((iso_days >= s) & (iso_days <= e)).any())))
    validation = pd.DataFrame(validation)

    _plot(series["Total"].reindex(days, fill_value=0), expected["Total"], incidents, days, cfg)
    if len(incidents):
        save_table(incidents.round(2).assign(start=incidents["start"].dt.date, end=incidents["end"].dt.date),
                   cfg.tables_dir / "anomaly_incidents.csv")
    save_table(validation, cfg.tables_dir / "anomaly_validation.csv")
    save_table(pd.DataFrame({"date": iso_days.date}), cfg.tables_dir / "anomaly_isolation_forest_days.csv")

    top = incidents.head(10).copy() if len(incidents) else pd.DataFrame()
    out = dict(series_monitored=len(series), flagged_days=int(len(flags)), incidents=int(len(incidents)),
               raw_incidents=int(len(all_incidents)), materiality_threshold=float(min_impact),
               isolation_forest_days=int(len(iso_days)),
               detected_injected=int(validation["detected"].sum()) if len(validation) else None,
               injected_total=int(len(validation)) if len(validation) else None,
               validation=validation.to_dict("records"),
               top_incidents=top.assign(start=top["start"].dt.strftime("%Y-%m-%d"),
                                        end=top["end"].dt.strftime("%Y-%m-%d")).round(2).to_dict("records")
               if len(top) else [],
               lost_revenue_from_drops=float(incidents.loc[incidents["direction"] == "drop", "impact"].sum())
               if len(incidents) else 0.0)
    # daily totals for the dashboard
    save_table(pd.DataFrame({"date": days.date, "actual": series["Total"].reindex(days, fill_value=0).round(2),
                             "expected": np.round(expected["Total"], 2), "z": np.round(resid["Total"], 3)}),
               cfg.tables_dir / "anomaly_daily_total.csv")
    save_json(out, cfg.reports_dir / "results" / "anomaly.json")
    log.info("%d incidents; injected detected %s/%s", len(incidents), out["detected_injected"], out["injected_total"])
    return out


def _plot(total, expected, incidents, days, cfg):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(days, expected, color=viz.NEUTRAL, lw=1.2, label="Expected (out-of-fold model)")
    ax.plot(days, total.to_numpy(), color=viz.PALETTE[0], lw=0.9, label="Actual daily revenue")
    if len(incidents):
        big = incidents.head(8)
        for r in big.itertuples(index=False):
            mid = r.start + (r.end - r.start) / 2
            col = viz.STATUS["critical"] if r.direction == "drop" else viz.STATUS["good"]
            ax.axvspan(r.start - pd.Timedelta(hours=12), r.end + pd.Timedelta(hours=12), color=col, alpha=0.15, lw=0)
            y = total.loc[r.start:r.end].max() if r.direction == "spike" else total.loc[r.start:r.end].min()
            label = r.series.split(": ")[-1]
            ax.annotate(f"{'▼' if r.direction == 'drop' else '▲'} {label}\n{r.start:%d %b %Y}", (mid, y),
                        xytext=(0, 18 if r.direction == "spike" else -26), textcoords="offset points",
                        ha="center", fontsize=7.5, color=viz.INK_2)
    viz.money_axis(ax)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=2)
    viz.titled(ax, "Daily revenue vs expected, with detected incidents",
               "▼ unexplained drop   ▲ unexplained spike (largest 8 incidents across all monitored series)")
    viz.save(fig, cfg.figures_dir / "16_anomaly_detection.png")
