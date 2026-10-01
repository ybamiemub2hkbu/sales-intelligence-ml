"""Promotion effectiveness: incremental profit and ROI per event.

Raw "sales during the promo" massively overstates promotions - most of those
customers would have bought anyway.  We estimate the **counterfactual**: a
model trained only on category-days that were *not* on promotion (and not in
a site-wide event or its aftermath) predicts what each category would have
sold on its promo days.  Then, per event:

* incremental net revenue      = actual - counterfactual
* incremental gross profit     = actual GP - counterfactual revenue x normal margin
* post-event dip               = the same comparison for the 10 days after
* ROI                          = (incremental GP incl. dip) / discount given away

Halo effects on non-promoted categories are *not* claimed: those category-days
are part of the training data, so any halo is absorbed into the baseline
(a conservative choice - promotions are, if anything, under-credited).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

from .. import viz
from ..config import Config
from ..utils import get_logger, save_json, save_table
from .anomaly import known_holidays

log = get_logger(__name__)
POST_DAYS = 10


def _cat_day_frame(tx, days, categories):
    g = tx.groupby(["order_date", "category"]).agg(net=("net_revenue", "sum"), gp=("gross_profit", "sum"),
                                                   disc=("discount_amount", "sum"))
    idx = pd.MultiIndex.from_product([days, categories], names=["order_date", "category"])
    return g.reindex(idx, fill_value=0).reset_index()


def _features(df, days):
    from ..data.generator import thanksgiving
    d = df["order_date"]
    hol = pd.Series(known_holidays(days), index=days)
    bf = {y: thanksgiving(y) + pd.Timedelta(days=1) for y in d.dt.year.unique()}
    return pd.DataFrame({
        "cat": pd.Categorical(df["category"], categories=viz.CATEGORIES).codes,
        "dow": d.dt.dayofweek, "doy_sin": np.sin(2 * np.pi * d.dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d.dt.dayofyear / 365.25), "month": d.dt.month,
        "trend": (d - days[0]).dt.days, "holiday": hol.reindex(d).to_numpy().astype(float),
        "pre_christmas": ((d.dt.month == 12) & d.dt.day.between(8, 23)).astype(float),
        "days_to_christmas": (d - pd.to_datetime(d.dt.year.astype(str) + "-12-25")).dt.days,
        "days_to_black_friday": (d - d.dt.year.map(bf)).dt.days.clip(-40, 40),
    })


def run(data: dict, cfg: Config) -> dict:
    tx, promos = data["transactions"], data["promotions"]
    days = pd.date_range(tx["order_date"].min(), tx["order_date"].max(), freq="D")
    cats = viz.CATEGORIES
    df = _cat_day_frame(tx, days, cats)

    # a category-day is "clean" if that category is not promoted and no site-wide
    # event (or its post-event dip window) is running
    on_promo = pd.DataFrame(False, index=days, columns=cats)
    sitewide = pd.Series(False, index=days)
    past = promos[promos["start_date"] <= days[-1]]
    for p in past.itertuples(index=False):
        m = (days >= p.start_date) & (days <= p.end_date)
        if p.categories == "ALL":
            after = (days > p.end_date) & (days <= p.end_date + pd.Timedelta(days=POST_DAYS))
            sitewide[m | after] = True
        else:
            for c in p.categories.split("|"):
                on_promo.loc[m, c] = True
    stacked = on_promo.stack()
    promo_flag = stacked.reindex(pd.MultiIndex.from_arrays([df["order_date"], df["category"]])).to_numpy()
    clean = ~promo_flag & ~sitewide.reindex(df["order_date"]).to_numpy()

    X = _features(df, days)
    model = HistGradientBoostingRegressor(max_iter=300 if cfg.fast else 600, learning_rate=0.04,
                                          max_leaf_nodes=31, min_samples_leaf=20, categorical_features=[0],
                                          random_state=cfg.seed)
    model.fit(X[clean], np.log1p(df.loc[clean, "net"]))
    df["cf_net"] = np.expm1(model.predict(X))
    # small-sample retransformation bias: rescale so clean days are unbiased on average
    adj = df.loc[clean, "net"].sum() / df.loc[clean, "cf_net"].sum()
    df["cf_net"] *= adj
    normal_margin = (df[clean].groupby("category")["gp"].sum() / df[clean].groupby("category")["net"].sum())
    df["cf_gp"] = df["cf_net"] * df["category"].map(normal_margin)
    log.info("counterfactual model trained on %d clean category-days (bias adj %.3f)", clean.sum(), adj)

    rows = []
    for p in past.itertuples(index=False):
        pc = cats if p.categories == "ALL" else p.categories.split("|")
        during = (df["order_date"] >= p.start_date) & (df["order_date"] <= p.end_date)
        after = (df["order_date"] > p.end_date) & (df["order_date"] <= p.end_date + pd.Timedelta(days=POST_DAYS))
        inc = df["category"].isin(pc)
        d, a = df[during & inc], df[after & inc]
        incr_gp_during = d["gp"].sum() - d["cf_gp"].sum()
        post_gp = a["gp"].sum() - a["cf_gp"].sum()
        discount = d["disc"].sum()
        total_incr_gp = incr_gp_during + post_gp
        rows.append(dict(
            promo_id=p.promo_id, promo_name=p.promo_name, year=p.start_date.year,
            start=p.start_date.date(), end=p.end_date.date(), days=(p.end_date - p.start_date).days + 1,
            scope=p.categories.replace("|", ", "), discount_pct=p.discount_pct,
            actual_net=d["net"].sum(), counterfactual_net=d["cf_net"].sum(),
            uplift_pct=d["net"].sum() / max(d["cf_net"].sum(), 1e-9) - 1,
            incremental_revenue=d["net"].sum() - d["cf_net"].sum(),
            discount_cost=discount, incremental_gp_during=incr_gp_during,
            post_event_gp=post_gp, total_incremental_gp=total_incr_gp,
            roi=total_incr_gp / max(discount, 1e-9),
        ))
    events = pd.DataFrame(rows)
    summary = (events.groupby("promo_name")
               .agg(runs=("promo_id", "size"), discount_pct=("discount_pct", "first"), scope=("scope", "first"),
                    avg_uplift_pct=("uplift_pct", "mean"), incremental_revenue=("incremental_revenue", "sum"),
                    discount_cost=("discount_cost", "sum"), post_event_gp=("post_event_gp", "sum"),
                    total_incremental_gp=("total_incremental_gp", "sum")))
    summary["roi"] = summary["total_incremental_gp"] / summary["discount_cost"]
    summary["verdict"] = np.select([summary["roi"] >= 0.5, summary["roi"] >= 0], ["Scale", "Optimise"], "Redesign / stop")
    summary = summary.sort_values("roi", ascending=False)

    # planned promotions for next year, annotated with the historical verdict
    planned = promos[promos["start_date"] > days[-1]][["promo_id", "promo_name", "start_date", "end_date",
                                                         "discount_pct", "categories"]].copy()
    planned = planned.join(summary[["roi", "verdict"]], on="promo_name")

    _plot(summary, events, cfg)
    save_table(events.round(4), cfg.tables_dir / "promotion_events.csv")
    save_table(summary.round(4).reset_index(), cfg.tables_dir / "promotion_summary.csv")
    save_table(planned.round(4), cfg.tables_dir / "promotion_plan_next_year_review.csv")

    out = dict(
        events=len(events), total_discount_cost=float(events["discount_cost"].sum()),
        total_incremental_gp=float(events["total_incremental_gp"].sum()),
        overall_roi=float(events["total_incremental_gp"].sum() / events["discount_cost"].sum()),
        naive_promo_revenue=float(events["actual_net"].sum()),
        incremental_share_of_promo_revenue=float(events["incremental_revenue"].sum() / events["actual_net"].sum()),
        post_event_gp_total=float(events["post_event_gp"].sum()),
        summary=summary.round(4).reset_index().to_dict("records"),
        best=summary.index[0], worst=summary.index[-1],
        planned_to_review=planned.loc[planned["verdict"] == "Redesign / stop", "promo_name"].tolist(),
    )
    save_json(out, cfg.reports_dir / "results" / "promotions.json")
    log.info("promo ROI overall %.2f; best %s, worst %s", out["overall_roi"], out["best"], out["worst"])
    return out


def _plot(summary, events, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    ax = axes[0]
    s = summary.sort_values("total_incremental_gp")
    colors = [viz.PALETTE[0] if v >= 0 else viz.PALETTE[7] for v in s["total_incremental_gp"]]
    ax.barh(s.index, s["total_incremental_gp"], color=colors, height=0.62)
    ax.axvline(0, color=viz.AXIS, lw=1)
    for i, (v, roi) in enumerate(zip(s["total_incremental_gp"], s["roi"])):
        ax.text(v, i, f"  ROI {roi:+.2f}  " if v >= 0 else f"ROI {roi:+.2f}  ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.3, color=viz.INK_2)
    lo, hi = s["total_incremental_gp"].min(), s["total_incremental_gp"].max()
    pad = (hi - lo) * 0.25
    ax.set_xlim(min(lo - pad, -pad * 0.3), hi + pad)
    viz.money_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    viz.titled(ax, "Incremental gross profit by promotion (all runs)",
               "vs counterfactual, including the post-event dip. ROI = incremental GP / discount given")

    ax = axes[1]
    ax.scatter(events["discount_pct"], events["uplift_pct"], s=np.clip(events["discount_cost"] / 150, 15, 400),
               color=viz.PALETTE[0], alpha=0.55, edgecolor=viz.SURFACE, linewidth=1.5)
    for r in summary.itertuples():
        e = events[events["promo_name"] == r.Index]
        ax.annotate(r.Index, (e["discount_pct"].mean(), e["uplift_pct"].mean()), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.4, color=viz.INK_2)
    viz.pct_axis(ax)
    viz.pct_axis(ax, "x")
    ax.set_xlabel("Discount depth")
    ax.set_ylabel("Revenue uplift vs counterfactual")
    viz.titled(ax, "Deeper discounts buy volume, not necessarily profit", "Each dot = one event; size = discount cost")
    viz.save(fig, cfg.figures_dir / "17_promotion_roi.png")
