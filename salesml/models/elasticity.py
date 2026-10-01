"""Price elasticity of demand and guard-railed price recommendations.

Identification
--------------
For each category we regress weekly log units of each product on its log
effective price with **product fixed effects** (absorb popularity) and
**week fixed effects** (absorb traffic, seasonality and category-wide
promotions).  What is left is within-week, cross-product price variation from
list-price changes - close to a natural experiment.  Uncertainty comes from a
week-clustered bootstrap.

Decision
--------
With constant elasticity e, profit (p - c) * q0 * (p / p0)^e is maximised at
p* = c * e / (1 + e) when e < -1.  Recommendations are clipped to a +/-
guardrail around today's price because the model is only trusted near the
prices it has observed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import viz
from ..config import Config
from ..utils import get_logger, load_json, save_json, save_table

log = get_logger(__name__)


def weekly_product_panel(tx: pd.DataFrame) -> pd.DataFrame:
    g = (tx.groupby(["category", "product_id", "week_start"])
         .agg(units=("quantity", "sum"), revenue=("revenue", "sum"))
         .reset_index())
    g = g[g["units"] > 0]
    g["price"] = g["revenue"] / g["units"]
    g["log_q"] = np.log(g["units"])
    g["log_p"] = np.log(g["price"])
    return g


def _within_estimate(d: pd.DataFrame) -> float:
    """Two-way fixed-effects slope via alternating demeaning (fast, exact at convergence)."""
    y, x = d["log_q"].to_numpy().copy(), d["log_p"].to_numpy().copy()
    prod = pd.factorize(d["product_id"])[0]
    week = pd.factorize(d["week_start"])[0]
    for _ in range(30):
        for codes in (prod, week):
            n = np.bincount(codes)
            y -= (np.bincount(codes, y) / n)[codes]
            x -= (np.bincount(codes, x) / n)[codes]
    denom = (x * x).sum()
    return float((x * y).sum() / denom) if denom > 0 else np.nan


def estimate_elasticities(panel: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cat, d in panel.groupby("category"):
        est = _within_estimate(d)
        groups = [g for _, g in d.groupby("week_start")]
        boots = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), size=len(groups))
            # relabel so a week drawn twice gets its own fixed effect each time
            parts = [groups[j].assign(week_start=i) for i, j in enumerate(pick)]
            boots.append(_within_estimate(pd.concat(parts)))
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        rows.append(dict(category=cat, elasticity=est, ci_low=lo, ci_high=hi, n_obs=len(d)))
    return pd.DataFrame(rows).sort_values("elasticity")


def price_recommendations(tx, products, elast, guardrail) -> pd.DataFrame:
    recent = tx[tx["order_date"] > tx["order_date"].max() - pd.Timedelta(weeks=12)]
    q0 = recent.groupby("product_id")["quantity"].sum() / 12
    p = products.set_index("product_id").join(q0.rename("weekly_units")).fillna({"weekly_units": 0})
    p = p.join(elast.set_index("category")["elasticity"], on="category")
    p0, c, e = p["current_list_price"], p["unit_cost"], p["elasticity"]
    unconstrained = np.where(e < -1, c * e / (1 + e), p0 * (1 + guardrail))
    p["optimal_price_unconstrained"] = unconstrained
    p["recommended_price"] = np.clip(unconstrained, p0 * (1 - guardrail), p0 * (1 + guardrail))
    # psychological price ending
    p["recommended_price"] = np.where(p["recommended_price"] > 5, np.round(p["recommended_price"]) - 0.01,
                                      p["recommended_price"].round(2))
    ratio = p["recommended_price"] / p0
    q1 = p["weekly_units"] * ratio ** e
    p["price_change_pct"] = ratio - 1
    p["weekly_profit_now"] = (p0 - c) * p["weekly_units"]
    p["weekly_profit_new"] = (p["recommended_price"] - c) * q1
    p["weekly_revenue_now"] = p0 * p["weekly_units"]
    p["weekly_revenue_new"] = p["recommended_price"] * q1
    p["annual_profit_uplift"] = (p["weekly_profit_new"] - p["weekly_profit_now"]) * 52
    p["margin_now"] = 1 - c / p0
    p["action"] = np.select([p["price_change_pct"] > 0.005, p["price_change_pct"] < -0.005],
                            ["Raise", "Lower"], "Hold")
    return p.reset_index()


def run(data: dict, cfg: Config) -> dict:
    tx, products = data["transactions"], data["products"]
    panel = weekly_product_panel(tx)
    elast = estimate_elasticities(panel, n_boot=60 if cfg.fast else 200, seed=cfg.seed)

    gt_path = cfg.raw_dir / "ground_truth.json"
    if gt_path.exists():
        gt = load_json(gt_path)["category_elasticity"]
        elast["true_elasticity_simulated"] = elast["category"].map(gt)
        elast["ci_covers_truth"] = (elast["true_elasticity_simulated"].between(elast["ci_low"], elast["ci_high"]))
    recs = price_recommendations(tx, products, elast, cfg.price_change_guardrail)
    by_cat = (recs.groupby("category")
              .agg(products=("product_id", "size"), raise_=("action", lambda s: (s == "Raise").sum()),
                   lower=("action", lambda s: (s == "Lower").sum()),
                   avg_price_change=("price_change_pct", "mean"),
                   annual_profit_uplift=("annual_profit_uplift", "sum"),
                   weekly_profit_now=("weekly_profit_now", "sum"),
                   weekly_revenue_now=("weekly_revenue_now", "sum"),
                   weekly_revenue_new=("weekly_revenue_new", "sum"))
              .rename(columns={"raise_": "raise"}))
    by_cat["profit_uplift_pct"] = by_cat["annual_profit_uplift"] / (by_cat["weekly_profit_now"] * 52)
    by_cat["revenue_change_pct"] = by_cat["weekly_revenue_new"] / by_cat["weekly_revenue_now"] - 1
    by_cat = by_cat.join(elast.set_index("category")["elasticity"]).sort_values("annual_profit_uplift", ascending=False)

    _plot(elast, by_cat, cfg)
    save_table(elast.round(4), cfg.tables_dir / "price_elasticity_by_category.csv")
    cols = ["product_id", "product_name", "category", "current_list_price", "unit_cost", "margin_now",
            "elasticity", "recommended_price", "price_change_pct", "action", "weekly_units",
            "annual_profit_uplift"]
    save_table(recs[cols].sort_values("annual_profit_uplift", ascending=False).round(4),
               cfg.tables_dir / "price_recommendations.csv")
    save_table(by_cat.round(4).reset_index(), cfg.tables_dir / "pricing_impact_by_category.csv")

    out = dict(
        elasticities=elast.round(4).to_dict("records"),
        guardrail=cfg.price_change_guardrail,
        total_annual_profit_uplift=float(recs["annual_profit_uplift"].sum()),
        current_annual_profit=float(recs["weekly_profit_now"].sum() * 52),
        n_raise=int((recs["action"] == "Raise").sum()), n_lower=int((recs["action"] == "Lower").sum()),
        n_hold=int((recs["action"] == "Hold").sum()),
        by_category=by_cat.round(4).reset_index().to_dict("records"),
        truth_coverage=float(elast["ci_covers_truth"].mean()) if "ci_covers_truth" in elast else None,
        mean_abs_error_vs_truth=float((elast["elasticity"] - elast["true_elasticity_simulated"]).abs().mean())
        if "true_elasticity_simulated" in elast else None,
        top_products=recs.nlargest(8, "annual_profit_uplift")[
            ["product_name", "category", "current_list_price", "recommended_price", "annual_profit_uplift"]
        ].round(2).to_dict("records"),
    )
    save_json(out, cfg.reports_dir / "results" / "elasticity.json")
    log.info("elasticities: %s", dict(zip(elast["category"], elast["elasticity"].round(2))))
    return out


def _plot(elast, by_cat, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    e = elast.sort_values("elasticity", ascending=False).reset_index(drop=True)
    y = np.arange(len(e))
    ax.hlines(y, e["ci_low"], e["ci_high"], color=viz.PALETTE[0], lw=2, alpha=0.45)
    ax.scatter(e["elasticity"], y, color=viz.PALETTE[0], s=46, zorder=3, edgecolor=viz.SURFACE, linewidth=1.5,
               label="Estimate with 95% CI")
    if "true_elasticity_simulated" in e:
        ax.scatter(e["true_elasticity_simulated"], y, marker="|", s=160, color=viz.INK, zorder=4,
                   label="Simulation ground truth")
    ax.axvline(-1, color=viz.AXIS, lw=1)
    ax.text(-1, len(e) - 0.45, " unit elastic", fontsize=8, color=viz.MUTED, va="bottom")
    ax.set_yticks(y, e["category"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Price elasticity (% change in units for +1% price)")
    ax.legend(loc="lower left")
    viz.titled(ax, "Price elasticity by category", "Two-way fixed effects, week-clustered bootstrap")

    ax = axes[1]
    b = by_cat.sort_values("annual_profit_uplift")
    ax.barh(b.index, b["annual_profit_uplift"], color=viz.PALETTE[0], height=0.62)
    for i, (v, pc) in enumerate(zip(b["annual_profit_uplift"], b["avg_price_change"])):
        ax.text(v, i, f"  ${v / 1e3:,.1f}K  (avg price {pc:+.0%})", va="center", fontsize=8.5, color=viz.INK_2)
    ax.set_xlim(min(0, b["annual_profit_uplift"].min() * 1.1), b["annual_profit_uplift"].max() * 1.65)
    viz.money_axis(ax, "x")
    ax.grid(axis="y", visible=False)
    viz.titled(ax, "Annual gross-profit uplift from repricing",
               f"Recommendations limited to +/-{cfg.price_change_guardrail:.0%} of current list price")
    viz.save(fig, cfg.figures_dir / "14_price_elasticity.png")
