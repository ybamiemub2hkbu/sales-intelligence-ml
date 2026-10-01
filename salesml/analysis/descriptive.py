"""Descriptive analytics: what happened, where, and for whom.

Produces the KPI block and the 'state of the business' figures used at the top
of the executive report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import viz
from ..config import Config
from ..utils import get_logger, save_json, save_table

log = get_logger(__name__)


def compute_kpis(tx: pd.DataFrame) -> dict:
    end = tx["order_date"].max()
    last_year = end.year
    yr = tx.groupby(tx["order_date"].dt.year)["net_revenue"].sum()
    active = tx.loc[tx["order_date"] > end - pd.Timedelta(days=365), "customer_id"].nunique()
    orders = tx.groupby("order_id")["net_revenue"].sum()
    return dict(
        period_start=tx["order_date"].min(), period_end=end,
        net_revenue=float(tx["net_revenue"].sum()),
        gross_revenue=float(tx["revenue"].sum()),
        gross_profit=float(tx["gross_profit"].sum()),
        gross_margin=float(tx["gross_profit"].sum() / tx["net_revenue"].sum()),
        orders=int(tx["order_id"].nunique()),
        customers=int(tx["customer_id"].nunique()),
        active_customers_365d=int(active),
        avg_order_value=float(orders.mean()),
        return_rate_revenue=float(tx.loc[tx["is_returned"] == 1, "revenue"].sum() / tx["revenue"].sum()),
        revenue_lost_to_returns=float(tx.loc[tx["is_returned"] == 1, "revenue"].sum()),
        discount_spend=float(tx["discount_amount"].sum()),
        revenue_by_year={int(k): float(v) for k, v in yr.items()},
        yoy_growth_last_year=float(yr[last_year] / yr[last_year - 1] - 1) if last_year - 1 in yr else None,
    )


def _monthly_trend(tx, cfg):
    m = tx.groupby(["month"])["net_revenue"].sum().reset_index()
    m["year"], m["mo"] = m["month"].dt.year, m["month"].dt.month
    years = sorted(m["year"].unique())
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for y in years:
        d = m[m["year"] == y]
        latest = y == years[-1]
        ax.plot(d["mo"], d["net_revenue"], color=viz.PALETTE[0] if latest else viz.NEUTRAL,
                lw=2.4 if latest else 1.6, marker="o" if latest else None, ms=4, zorder=3 if latest else 2)
        ax.annotate(str(y), (d["mo"].iloc[-1], d["net_revenue"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=viz.INK if latest else viz.INK_2, fontweight="semibold" if latest else "normal")
    ax.set_xticks(range(1, 13), ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlim(0.7, 12.8)
    ax.set_ylim(bottom=0)
    viz.money_axis(ax)
    yr = m.groupby("year")["net_revenue"].sum()
    growth = yr.iloc[-1] / yr.iloc[-2] - 1
    viz.titled(ax, "Monthly net revenue, year over year",
               f"{years[-1]} highlighted. Full-year growth vs {years[-2]}: {growth:+.1%}. Peaks follow Black Friday and the holidays.")
    return viz.save(fig, cfg.figures_dir / "01_monthly_revenue_yoy.png")


def _category_performance(tx, cfg):
    c = tx.groupby("category").agg(net=("net_revenue", "sum"), gp=("gross_profit", "sum"),
                                   disc=("discount_amount", "sum"), gross=("revenue", "sum"))
    c["margin"] = c["gp"] / c["net"]
    c = c.sort_values("net")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    axes[0].barh(c.index, c["net"], color=viz.PALETTE[0], height=0.62)
    viz.money_axis(axes[0], "x")
    viz.titled(axes[0], "Net revenue by category", "Three-year total after returns")
    axes[1].barh(c.index, c["margin"], color=viz.PALETTE[0], height=0.62)
    viz.pct_axis(axes[1], "x")
    viz.titled(axes[1], "Gross margin by category", "Gross profit / net revenue")
    for ax, col, fmt in ((axes[0], "net", lambda v: f"${v / 1e6:.2f}M"), (axes[1], "margin", lambda v: f"{v:.0%}")):
        for i, v in enumerate(c[col]):
            ax.text(v, i, " " + fmt(v), va="center", fontsize=8.5, color=viz.INK_2)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, c[col].max() * 1.22)
    viz.save(fig, cfg.figures_dir / "02_category_performance.png")
    return c.sort_values("net", ascending=False)


def _region_channel(tx, cfg):
    pv = tx.pivot_table(index="region", columns="channel", values="net_revenue", aggfunc="sum")
    pv = pv.loc[["North", "East", "South", "West"], ["In-Store", "Online", "Mobile App"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    im = ax.imshow(pv.values, cmap=viz.SEQ_CMAP, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    ax.grid(False)
    vmax = pv.values.max()
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"${v / 1e6:.2f}M", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.6 * vmax else viz.INK)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    viz.money_axis(cb.ax)
    viz.titled(ax, "Net revenue by region and channel", "Three-year total")
    viz.save(fig, cfg.figures_dir / "03_region_channel_heatmap.png")
    return pv


def _channel_mix(tx, cfg):
    q = tx.assign(q=tx["order_date"].dt.to_period("Q").astype(str))
    mix = q.pivot_table(index="q", columns="channel", values="net_revenue", aggfunc="sum")
    mix = mix[["In-Store", "Online", "Mobile App"]]
    share = mix.div(mix.sum(1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bottom = np.zeros(len(share))
    x = np.arange(len(share))
    for ch in share.columns:
        ax.bar(x, share[ch], bottom=bottom, color=viz.CHANNEL_COLORS[ch], width=0.78,
               edgecolor=viz.SURFACE, linewidth=1.5, label=ch)
        ax.text(x[-1] + 0.5, bottom[-1] + share[ch].iloc[-1] / 2, f"{ch} {share[ch].iloc[-1]:.0%}",
                va="center", fontsize=8.5, color=viz.INK_2)
        bottom += share[ch].to_numpy()
    ax.set_xticks(x, share.index, rotation=0, fontsize=7.8)
    ax.set_xlim(-0.6, len(x) + 1.6)
    viz.pct_axis(ax)
    ax.set_ylim(0, 1)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.08))
    viz.titled(ax, "Channel mix by quarter",
               f"Mobile App share moved from {share['Mobile App'].iloc[0]:.0%} to {share['Mobile App'].iloc[-1]:.0%} of net revenue")
    viz.save(fig, cfg.figures_dir / "04_channel_mix.png")
    return share


def _cohort_retention(tx, cfg):
    first = tx.groupby("customer_id")["order_date"].min().dt.to_period("Q")
    t = tx[["customer_id", "order_date"]].drop_duplicates()
    t = t.assign(cohort=t["customer_id"].map(first), q=t["order_date"].dt.to_period("Q"))
    t["age"] = (t["q"] - t["cohort"]).apply(lambda d: d.n)
    size = t[t["age"] == 0].groupby("cohort")["customer_id"].nunique()
    ret = t.groupby(["cohort", "age"])["customer_id"].nunique().unstack()
    ret = ret.div(size, axis=0)
    # first-quarter cohort is polluted by pre-existing customers; drop it
    ret = ret.iloc[1:]
    later = ret.iloc[:, 1:].dropna(how="all").dropna(axis=1, how="all")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    data = later.to_numpy(dtype=float)
    ret = ret.loc[later.index]
    im = ax.imshow(np.ma.masked_invalid(data), cmap=viz.SEQ_CMAP, aspect="auto",
                   vmin=np.nanmin(data) * 0.7, vmax=np.nanmax(data))
    ax.set_xticks(range(data.shape[1]), [f"Q+{i}" for i in range(1, data.shape[1] + 1)])
    ax.set_yticks(range(len(ret)), [str(c) for c in ret.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                rel = (data[i, j] - im.norm.vmin) / (im.norm.vmax - im.norm.vmin)
                ax.text(j, i, f"{data[i, j]:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if rel > 0.55 else viz.INK)
    ax.set_xlabel("Quarters after first purchase")
    ax.set_ylabel("Acquisition cohort")
    q1 = np.nanmean(data[:, 0])
    viz.titled(ax, "Cohort retention: share of each cohort buying again",
               f"On average {q1:.0%} of new customers return in the next quarter")
    viz.save(fig, cfg.figures_dir / "05_cohort_retention.png")
    return ret


def _pareto(tx, cfg):
    cust = tx.groupby("customer_id")["net_revenue"].sum().sort_values(ascending=False)
    cum = cust.cumsum() / cust.sum()
    x = np.arange(1, len(cust) + 1) / len(cust)
    top20 = float(cum.iloc[int(len(cust) * 0.2) - 1])
    top10 = float(cum.iloc[int(len(cust) * 0.1) - 1])
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(x, cum.to_numpy(), color=viz.PALETTE[0])
    ax.plot([0, 1], [0, 1], color=viz.NEUTRAL, lw=1)
    ax.scatter([0.2], [top20], s=40, color=viz.PALETTE[0], edgecolor=viz.SURFACE, linewidth=2, zorder=4)
    ax.annotate(f"Top 20% of customers\n= {top20:.0%} of revenue", (0.2, top20), xytext=(18, -30),
                textcoords="offset points", fontsize=9, color=viz.INK)
    viz.pct_axis(ax)
    viz.pct_axis(ax, "x")
    ax.set_xlabel("Share of customers (ranked by spend)")
    ax.set_ylabel("Cumulative share of net revenue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    viz.titled(ax, "Revenue concentration", "Diagonal = every customer spends the same")
    viz.save(fig, cfg.figures_dir / "06_revenue_concentration.png")
    return dict(top10_share=top10, top20_share=top20)


def _returns(tx, cfg):
    r = tx.pivot_table(index="category", columns="channel", values="is_returned", aggfunc="mean")
    r = r[["In-Store", "Online", "Mobile App"]].sort_values("Online", ascending=False)
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    im = ax.imshow(r.values, cmap=viz.SEQ_CMAP, aspect="auto", vmin=0)
    ax.set_xticks(range(r.shape[1]), r.columns)
    ax.set_yticks(range(r.shape[0]), r.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.values[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if v > 0.6 * r.values.max() else viz.INK)
    viz.titled(ax, "Return rate by category and channel", "Share of order lines returned")
    viz.save(fig, cfg.figures_dir / "07_return_rates.png")
    return r


def abc_products(tx, products):
    p = tx.groupby("product_id").agg(net_revenue=("net_revenue", "sum"), gross_profit=("gross_profit", "sum"),
                                     units=("quantity", "sum")).sort_values("net_revenue", ascending=False)
    p["cum_share"] = p["net_revenue"].cumsum() / p["net_revenue"].sum()
    p["abc_class"] = np.select([p["cum_share"] <= 0.80, p["cum_share"] <= 0.95], ["A", "B"], "C")
    p = p.join(products.set_index("product_id")[["product_name", "category"]])
    return p.reset_index()


def run(data: dict, cfg: Config) -> dict:
    tx, products = data["transactions"], data["products"]
    kpis = compute_kpis(tx)
    cat = _category_performance(tx, cfg)
    _monthly_trend(tx, cfg)
    rc = _region_channel(tx, cfg)
    mix = _channel_mix(tx, cfg)
    coh = _cohort_retention(tx, cfg)
    conc = _pareto(tx, cfg)
    ret = _returns(tx, cfg)
    abc = abc_products(tx, products)

    save_table(cat.reset_index(), cfg.tables_dir / "category_performance.csv")
    save_table(abc, cfg.tables_dir / "product_abc_classification.csv")
    save_table(coh.reset_index().astype({"cohort": str}), cfg.tables_dir / "cohort_retention.csv")

    out = dict(kpis=kpis, concentration=conc,
               category=cat.reset_index().to_dict("records"),
               channel_share_first=mix.iloc[0].to_dict(), channel_share_last=mix.iloc[-1].to_dict(),
               region_channel=rc.to_dict(),
               avg_next_quarter_retention=float(np.nanmean(coh.iloc[:, 1])),
               return_hotspot=dict(category=ret["Online"].idxmax(), rate=float(ret["Online"].max()),
                                   instore_rate=float(ret.loc[ret["Online"].idxmax(), "In-Store"])),
               abc_counts=abc["abc_class"].value_counts().to_dict(),
               a_items_share_of_skus=float((abc["abc_class"] == "A").mean()))
    save_json(out, cfg.reports_dir / "results" / "descriptive.json")
    log.info("net revenue %.0f, margin %.1f%%, YoY %.1f%%", kpis["net_revenue"],
             100 * kpis["gross_margin"], 100 * (kpis["yoy_growth_last_year"] or 0))
    return out
