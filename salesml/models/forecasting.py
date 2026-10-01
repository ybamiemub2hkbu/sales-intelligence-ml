"""Weekly demand forecasting per category (next quarter).

Approach
--------
* One *global* gradient-boosted model across all categories (more data per
  model than eight separate ones, and categories share seasonal structure).
* Trees cannot extrapolate a growing trend, so the target is the log-ratio of
  the week's revenue to its trailing 13-week level.  The model learns the
  *shape* (seasonality, promotions, holidays); the level carries the trend.
* Last year's same-week seasonal ratio, the planned promotion calendar and
  holiday flags are the key drivers - all known in advance at forecast time.
* Multi-step forecasts are produced recursively.
* Accuracy is measured with rolling-origin backtests (each origin only sees
  its past) against two baselines planners actually use.
* Prediction intervals are empirical: quantiles of backtest errors by horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import viz
from ..config import Config
from ..data.generator import thanksgiving
from ..utils import bias, get_logger, save_json, save_table, wape

log = get_logger(__name__)
LEVEL_WINDOW = 13
MIN_HISTORY = 52 + LEVEL_WINDOW + 1


# ------------------------------------------------------------------ data --
def weekly_series(tx: pd.DataFrame) -> pd.DataFrame:
    """Category x week net revenue, complete weeks only."""
    w = tx.pivot_table(index="week_start", columns="category", values="net_revenue", aggfunc="sum").fillna(0)
    first, last = tx["order_date"].min(), tx["order_date"].max()
    full = (w.index >= first) & (w.index + pd.Timedelta(days=6) <= last)
    return w.loc[full].asfreq("7D").fillna(0)


def promo_calendar_weekly(promos: pd.DataFrame, weeks: pd.DatetimeIndex, categories: list[str]) -> dict:
    """Average category discount and site-wide promo days per week (known in advance)."""
    days = pd.date_range(weeks.min(), weeks.max() + pd.Timedelta(days=6), freq="D")
    disc = pd.DataFrame(0.0, index=days, columns=categories)
    sitewide = pd.Series(0.0, index=days)
    for p in promos.itertuples(index=False):
        m = (days >= p.start_date) & (days <= p.end_date)
        if not m.any():
            continue
        cats = categories if p.categories == "ALL" else [c for c in p.categories.split("|") if c in categories]
        for c in cats:
            disc.loc[m, c] = np.maximum(disc.loc[m, c], p.discount_pct)
        if p.categories == "ALL":
            sitewide[m] += 1
    wk = days - pd.to_timedelta(days.dayofweek, unit="D")
    return dict(disc=disc.groupby(wk).mean(), sitewide=sitewide.groupby(wk).sum())


def repair_incidents(series: pd.DataFrame, cal: dict, threshold: float = 0.30, z: float = 4.0):
    """Replace one-off dips/spikes that no promotion or holiday explains.

    A stock-out last February should not teach the model that February is
    weak.  Weeks deviating > ``threshold`` (log scale) from a centred 9-week
    median *and* > ``z`` robust standard deviations of that category's own
    week-to-week noise, with no promotion or holiday that week, are set to
    that median.
    Evaluation is always against the *unrepaired* actuals.
    """
    clean = series.copy()
    flags = []
    holiday = pd.Series({w: any(_calendar(w)[k] for k in ("has_christmas", "pre_christmas",
                                                          "has_thanksgiving", "has_new_year"))
                         for w in series.index})
    sitewide = cal["sitewide"].reindex(series.index).fillna(0) > 0
    for c in series.columns:
        y = series[c]
        med = y.rolling(9, center=True, min_periods=5).median()
        dev = np.log1p(y) - np.log1p(med)
        promo = cal["disc"][c].reindex(series.index).fillna(0) > 0
        scale = 1.4826 * (dev - dev.median()).abs().median()
        bad = (dev.abs() > threshold) & (dev.abs() > z * scale) & ~promo & ~sitewide & ~holiday
        clean.loc[bad, c] = med[bad]
        flags += [dict(category=c, week_start=w, actual=y[w], repaired_to=med[w]) for w in y.index[bad]]
    return clean, pd.DataFrame(flags)


def _calendar(week: pd.Timestamp) -> dict:
    days = pd.date_range(week, periods=7, freq="D")
    tg = thanksgiving(week.year)
    woy = int(week.isocalendar().week)
    return dict(
        week_of_year=woy, woy_sin=np.sin(2 * np.pi * woy / 52.18), woy_cos=np.cos(2 * np.pi * woy / 52.18),
        has_christmas=float(any((d.month == 12) & (d.day == 25) for d in days)),
        pre_christmas=float(any((d.month == 12) & (8 <= d.day <= 22) for d in days)),
        has_thanksgiving=float(tg in days), has_new_year=float(any((d.month == 1) & (d.day == 1) for d in days)),
    )


def _features(hist: np.ndarray, week: pd.Timestamp, cat_code: int, disc: float, sitewide: float,
              disc_ly: float, sitewide_ly: float) -> dict:
    """Features for the week right after ``hist`` (levels, not logs)."""
    t = len(hist)
    lvl = hist[t - LEVEL_WINDOW:t].mean()
    lvl_ly = hist[t - 52 - LEVEL_WINDOW:t - 52].mean()
    lg = lambda v: np.log1p(max(v, 0.0))  # noqa: E731
    f = dict(
        cat=cat_code,
        lag1_ratio=lg(hist[t - 1]) - lg(lvl), lag2_ratio=lg(hist[t - 2]) - lg(lvl),
        lag4_ratio=lg(hist[t - 4]) - lg(lvl),
        ly_ratio=lg(hist[t - 52]) - lg(lvl_ly),            # last year's same-week shape
        ly_next_ratio=lg(hist[t - 51]) - lg(lvl_ly),
        ly_prev_ratio=lg(hist[t - 53]) - lg(lvl_ly),
        momentum=lg(hist[t - 4:t].mean()) - lg(lvl),
        yoy_level=lg(lvl) - lg(lvl_ly),
        disc=disc, sitewide_days=sitewide, disc_ly=disc_ly, sitewide_ly=sitewide_ly,
    )
    f.update(_calendar(week))
    return f


class WeeklyForecaster:
    def __init__(self, series: pd.DataFrame, promo_cal: dict, seed: int = 42, fast: bool = False):
        self.series = series
        self.categories = list(series.columns)
        self.cal = promo_cal
        self.seed = seed
        self.fast = fast
        self.model = None

    def _promo(self, week, cat):
        ly = week - pd.Timedelta(weeks=52)
        get = lambda tab, w, c=None: float(tab.loc[w, c] if c else tab.loc[w]) if w in tab.index else 0.0  # noqa: E731
        return (get(self.cal["disc"], week, cat), get(self.cal["sitewide"], week),
                get(self.cal["disc"], ly, cat), get(self.cal["sitewide"], ly))

    def _training_rows(self, end_idx: int):
        X, y = [], []
        for ci, c in enumerate(self.categories):
            vals = self.series[c].to_numpy()
            for t in range(MIN_HISTORY, end_idx):
                week = self.series.index[t]
                f = _features(vals[:t], week, ci, *self._promo(week, c))
                lvl = vals[t - LEVEL_WINDOW:t].mean()
                X.append(f)
                y.append(np.log1p(vals[t]) - np.log1p(lvl))
        return pd.DataFrame(X), np.array(y)

    def fit(self, end_idx: int):
        X, y = self._training_rows(end_idx)
        self.model = HistGradientBoostingRegressor(
            max_iter=250 if self.fast else 500, learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=8, l2_regularization=1.0, categorical_features=[0],
            random_state=self.seed)
        self.model.fit(X, y)
        self.feature_names = list(X.columns)
        return self

    def predict(self, end_idx: int, horizon: int) -> pd.DataFrame:
        """Recursive forecast of ``horizon`` weeks after series index ``end_idx``."""
        start_week = self.series.index[0] + pd.Timedelta(weeks=end_idx)
        out = []
        for ci, c in enumerate(self.categories):
            hist = list(self.series[c].to_numpy()[:end_idx])
            for h in range(horizon):
                week = start_week + pd.Timedelta(weeks=h)
                f = _features(np.array(hist), week, ci, *self._promo(week, c))
                lvl = np.mean(hist[-LEVEL_WINDOW:])
                pred = float(np.expm1(self.model.predict(pd.DataFrame([f]))[0] + np.log1p(lvl)))
                pred = max(pred, 0.0)
                hist.append(pred)
                out.append(dict(category=c, week_start=week, horizon=h + 1, forecast=pred))
        return pd.DataFrame(out)


def seasonal_naive(series: pd.DataFrame, end_idx: int, horizon: int) -> pd.DataFrame:
    """Same week last year x recent YoY growth - a strong planner baseline."""
    out = []
    for c in series.columns:
        v = series[c].to_numpy()
        growth = v[end_idx - 13:end_idx].sum() / max(v[end_idx - 65:end_idx - 52].sum(), 1e-9)
        for h in range(horizon):
            t = end_idx + h
            out.append(dict(category=c, week_start=series.index[0] + pd.Timedelta(weeks=t),
                            horizon=h + 1, forecast=v[t - 52] * growth))
    return pd.DataFrame(out)


def moving_average(series: pd.DataFrame, end_idx: int, horizon: int, window: int = 8) -> pd.DataFrame:
    out = []
    for c in series.columns:
        m = series[c].to_numpy()[end_idx - window:end_idx].mean()
        for h in range(horizon):
            out.append(dict(category=c, week_start=series.index[0] + pd.Timedelta(weeks=end_idx + h),
                            horizon=h + 1, forecast=m))
    return pd.DataFrame(out)


# --------------------------------------------------------------- figures --
def _plot_backtest(series, bt, cfg, chosen):
    total = series.sum(1)
    agg = bt.groupby(["week_start", "method"])["forecast"].sum().unstack()
    start = agg.index.min() - pd.Timedelta(weeks=26)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    hist = total[total.index >= start]
    ax.plot(hist.index, hist.values, color=viz.INK, lw=1.6, label="Actual")
    ax.plot(agg.index, agg["Seasonal naive x growth"], color=viz.NEUTRAL, lw=1.6, label="Seasonal naive x growth (baseline)")
    ax.plot(agg.index, agg[chosen], color=viz.PALETTE[0], lw=2.2, label=chosen)
    origins = sorted(bt["origin"].unique())
    for o in origins:
        ax.axvline(o, color=viz.GRID, lw=1, zorder=0)
    viz.money_axis(ax)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=3)
    m_w = wape(bt.loc[bt.method == chosen, "actual"], bt.loc[bt.method == chosen, "forecast"])
    b_w = wape(bt.loc[bt.method == "Seasonal naive x growth", "actual"],
               bt.loc[bt.method == "Seasonal naive x growth", "forecast"])
    viz.titled(ax, "Backtest: weekly net revenue, all categories",
               f"{len(origins)} rolling 13-week origins (vertical rules). Category-level WAPE: {chosen.split(' (')[0]} {m_w:.1%} vs baseline {b_w:.1%}")
    viz.save(fig, cfg.figures_dir / "08_forecast_backtest.png")


def _plot_future(series, fut, cfg):
    cats = list(series.columns)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), sharex=True)
    for ax, c in zip(axes.flat, cats):
        h = series[c].iloc[-52:]
        f = fut[fut["category"] == c]
        col = viz.CATEGORY_COLORS[c]
        ax.plot(h.index, h.values, color=viz.INK_2, lw=1.3)
        ax.fill_between(f["week_start"], f["p10"], f["p90"], color=col, alpha=0.18, lw=0)
        ax.plot(f["week_start"], f["p50"], color=col, lw=2.2)
        ax.set_title(c, fontsize=10.5, pad=6)
        viz.money_axis(ax)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=0, labelsize=7.5)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    fig.suptitle("Next-quarter forecast by category (median and 80% interval)", x=0.01, ha="left",
                 fontsize=13, fontweight="semibold")
    fig.text(0.01, 0.925, "Grey: last 52 weeks actual. Coloured: forecast. Interval width comes from backtest errors at each horizon.",
             fontsize=9, color=viz.INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    viz.save(fig, cfg.figures_dir / "09_forecast_next_quarter.png")


# ------------------------------------------------------------------- run --
def run(data: dict, cfg: Config) -> dict:
    tx, promos = data["transactions"], data["promotions"]
    series = weekly_series(tx)
    H = cfg.forecast_horizon_weeks
    future_weeks = pd.date_range(series.index[-1] + pd.Timedelta(weeks=1), periods=H, freq="7D")
    cal = promo_calendar_weekly(promos, series.index.append(future_weeks), list(series.columns))
    T = len(series)
    n_orig = max(cfg.forecast_backtest_origins, 1)
    origins = [T - H * k for k in range(n_orig, 0, -1)]

    repaired, repairs = repair_incidents(series, cal)
    log.info("repaired %d unexplained category-weeks before modelling", len(repairs))
    fc = WeeklyForecaster(repaired, cal, seed=cfg.seed, fast=cfg.fast)
    bt = []
    for o in origins:
        fc.fit(o)
        preds = {"ML model": fc.predict(o, H), "Seasonal naive x growth": seasonal_naive(repaired, o, H),
                 "Moving average (8w)": moving_average(repaired, o, H)}
        preds["Ensemble (ML + seasonal naive)"] = preds["ML model"].assign(
            forecast=(preds["ML model"]["forecast"].to_numpy() + preds["Seasonal naive x growth"]["forecast"].to_numpy()) / 2)
        for name, p in preds.items():
            p = p.assign(method=name, origin=series.index[o])
            p["actual"] = [series.loc[w, c] for w, c in zip(p["week_start"], p["category"])]
            bt.append(p)
        log.info("backtest origin %s done", series.index[o].date())
    bt = pd.concat(bt, ignore_index=True)

    metrics = []
    for name, g in bt.groupby("method"):
        tot = g.groupby("week_start")[["actual", "forecast"]].sum()
        metrics.append(dict(method=name, wape_category_week=wape(g["actual"], g["forecast"]),
                            wape_total_week=wape(tot["actual"], tot["forecast"]),
                            bias=bias(g["actual"], g["forecast"])))
    metrics = pd.DataFrame(metrics).sort_values("wape_category_week")
    per_cat = (bt.groupby(["method", "category"]).apply(lambda g: wape(g["actual"], g["forecast"]))
               .unstack(0).round(4))

    # the production method is whichever won the backtest (ML or ensemble)
    candidates = metrics[metrics.method.isin(["ML model", "Ensemble (ML + seasonal naive)"])]
    chosen = candidates.iloc[0]["method"]

    # empirical interval: quantiles of log errors by horizon bucket
    ml = bt[bt["method"] == chosen].copy()
    ml["log_err"] = np.log1p(ml["actual"]) - np.log1p(ml["forecast"])
    ml["bucket"] = pd.cut(ml["horizon"], [0, 4, 8, 13], labels=["1-4", "5-8", "9-13"])
    q = ml.groupby("bucket", observed=True)["log_err"].quantile([0.1, 0.9]).unstack()

    # refit on all history and forecast the next quarter
    full = pd.concat([repaired, pd.DataFrame(0.0, index=future_weeks, columns=series.columns)])
    fc_full = WeeklyForecaster(full, cal, seed=cfg.seed, fast=cfg.fast).fit(T)
    fut = fc_full.predict(T, H).rename(columns={"forecast": "p50"})
    if chosen != "ML model":
        fut["p50"] = (fut["p50"].to_numpy() + seasonal_naive(full, T, H)["forecast"].to_numpy()) / 2
    b = pd.cut(fut["horizon"], [0, 4, 8, 13], labels=["1-4", "5-8", "9-13"])
    fut["p10"] = np.expm1(np.log1p(fut["p50"]) + b.map(q[0.1]).astype(float))
    fut["p90"] = np.expm1(np.log1p(fut["p50"]) + b.map(q[0.9]).astype(float))

    # compare with same quarter last year - repaired, so a past stock-out does not
    # masquerade as growth
    ly_weeks = future_weeks - pd.Timedelta(weeks=52)
    ly = repaired.loc[repaired.index.isin(ly_weeks)].sum()
    by_cat = fut.groupby("category")[["p10", "p50", "p90"]].sum()
    by_cat["same_weeks_last_year"] = ly
    by_cat["expected_growth"] = by_cat["p50"] / by_cat["same_weeks_last_year"] - 1
    by_cat = by_cat.sort_values("expected_growth", ascending=False)

    _plot_backtest(series, bt, cfg, chosen)
    _plot_future(series, fut, cfg)
    save_table(metrics, cfg.tables_dir / "forecast_backtest_metrics.csv")
    save_table(repairs.round(2), cfg.tables_dir / "forecast_repaired_weeks.csv")
    save_table(per_cat.reset_index(), cfg.tables_dir / "forecast_wape_by_category.csv")
    save_table(fut[["category", "week_start", "horizon", "p10", "p50", "p90"]].round(2),
               cfg.tables_dir / "forecast_next_quarter_weekly.csv")
    save_table(by_cat.round(4).reset_index(), cfg.tables_dir / "forecast_next_quarter_by_category.csv")

    imp = None
    try:
        from sklearn.inspection import permutation_importance
        Xf, yf = fc_full._training_rows(T)
        r = permutation_importance(fc_full.model, Xf, yf, n_repeats=5, random_state=cfg.seed)
        imp = pd.Series(r.importances_mean, index=Xf.columns).sort_values(ascending=False)
        save_table(imp.rename("importance").reset_index().rename(columns={"index": "feature"}),
                   cfg.tables_dir / "forecast_feature_importance.csv")
    except Exception as e:  # pragma: no cover - importance is a nice-to-have
        log.warning("feature importance skipped: %s", e)

    best = metrics.iloc[0]
    ml_row = metrics[metrics.method == "ML model"].iloc[0]
    base_row = metrics[metrics.method == "Seasonal naive x growth"].iloc[0]
    out = dict(
        horizon_weeks=H, n_origins=len(origins),
        metrics=metrics.to_dict("records"), best_method=best["method"], production_method=chosen,
        repaired_weeks=repairs.assign(week_start=repairs["week_start"].astype(str)).to_dict("records") if len(repairs) else [],
        production_wape=float(metrics[metrics.method == chosen]["wape_category_week"].iloc[0]),
        ml_wape=float(ml_row["wape_category_week"]), baseline_wape=float(base_row["wape_category_week"]),
        error_reduction_vs_baseline=float(1 - metrics[metrics.method == chosen]["wape_category_week"].iloc[0]
                                          / base_row["wape_category_week"]),
        next_quarter_total=dict(p10=float(by_cat["p10"].sum()), p50=float(by_cat["p50"].sum()),
                                p90=float(by_cat["p90"].sum()),
                                last_year=float(by_cat["same_weeks_last_year"].sum())),
        next_quarter_start=future_weeks[0], next_quarter_end=future_weeks[-1] + pd.Timedelta(days=6),
        by_category=by_cat.reset_index().to_dict("records"),
        top_features=list(imp.index[:6]) if imp is not None else [],
    )
    out["next_quarter_total"]["growth"] = out["next_quarter_total"]["p50"] / out["next_quarter_total"]["last_year"] - 1
    save_json(out, cfg.reports_dir / "results" / "forecasting.json")
    log.info("forecast WAPE %s %.3f vs seasonal-naive %.3f", chosen, out["production_wape"], out["baseline_wape"])
    return out
