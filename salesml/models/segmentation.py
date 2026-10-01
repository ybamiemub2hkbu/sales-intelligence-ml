"""Behavioural customer segmentation (RFM+ and K-Means).

Segments are built from *how* customers shop (recency, frequency, spend,
discount dependence, breadth, digital adoption, tenure), named automatically
from their profiles, and translated into a playbook of actions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .. import viz
from ..config import Config
from ..features import customer_snapshot
from ..utils import get_logger, save_json, save_table

log = get_logger(__name__)

# channel preference is deliberately *not* a clustering input: it is bimodal and
# would split customers by channel rather than by value and engagement.
SEG_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "discount_share",
                "n_categories", "tenure_days"]
PROFILE_FEATURES = SEG_FEATURES + ["digital_share", "return_rate"]
LAPSED_AFTER_DAYS = 365
LOG_FEATURES = ["recency_days", "frequency", "monetary", "avg_order_value", "tenure_days"]

PLAYBOOK = {
    "Champions": "Protect: VIP early access, loyalty tier upgrades, no blanket discounts needed.",
    "Steady Regulars": "Grow basket: cross-sell from basket rules, subscribe-and-save on consumables.",
    "Deal Hunters": "Monetise promos: target with event previews, steer to high-margin bundles, cap discount depth.",
    "New & Promising": "Second-purchase journey within 30 days: onboarding series + category recommendation.",
    "Occasional Shoppers": "Seasonal reminders tied to their categories; low-cost digital channels only.",
    "Cooling Down": "At risk: route to the churn model's retention list; personalised offer before they lapse.",
    "Lapsed / Hibernating": "One win-back attempt with a time-boxed offer; suppress from paid media if no response.",
}


def _rfm_scores(f: pd.DataFrame) -> pd.DataFrame:
    """Classic quintile RFM scores, kept alongside clusters for familiarity."""
    r = pd.qcut(f["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    fr = pd.qcut(f["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    m = pd.qcut(f["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    return pd.DataFrame({"R": r, "F": fr, "M": m, "RFM": r.astype(str) + fr.astype(str) + m.astype(str)})


def _name_segments(profile: pd.DataFrame) -> dict[int, str]:
    """Rule-based naming from cluster medians; every cluster gets a unique name."""
    names, left = {}, list(profile.index)

    def take(name, idx):
        names[idx] = name
        left.remove(idx)

    take("Champions", profile.loc[left, "monetary"].idxmax())
    if left:
        cand = profile.loc[left, "discount_share"].idxmax()
        if profile.loc[cand, "discount_share"] > profile["discount_share"].median() * 1.2:
            take("Deal Hunters", cand)
    if left:
        take("New & Promising", profile.loc[left, "tenure_days"].idxmin())
    if left:
        take("Steady Regulars", profile.loc[left, "frequency"].idxmax())
    if len(left) > 1:
        cooling = profile.loc[left, "recency_days"].idxmax()
        if profile.loc[cooling, "recency_days"] > 120:
            take("Cooling Down", cooling)
    for i, idx in enumerate(list(left)):
        take("Occasional Shoppers" if i == 0 else f"Occasional Shoppers {i + 1}", idx)
    return names


def run(data: dict, cfg: Config) -> dict:
    tx, customers = data["transactions"], data["customers"]
    as_of = tx["order_date"].max() + pd.Timedelta(days=1)
    f = customer_snapshot(tx, as_of, customers)
    # lapsed customers are a rule, not a cluster: they would dominate the geometry
    active = f["recency_days"] <= LAPSED_AFTER_DAYS
    X = f.loc[active, SEG_FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(X[c])
    Z = StandardScaler().fit_transform(X)

    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(len(Z), size=min(5000, len(Z)), replace=False)
    scores = {}
    for k in range(cfg.segment_k_range[0], cfg.segment_k_range[1] + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(Z)
        scores[k] = float(silhouette_score(Z[sample], km.labels_[sample]))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=20, random_state=cfg.seed).fit(Z)
    f["cluster"] = -1
    f.loc[active, "cluster"] = km.labels_
    log.info("silhouette by k: %s -> k=%d", {k: round(v, 3) for k, v in scores.items()}, best_k)

    profile = f[active].groupby("cluster")[SEG_FEATURES].median()
    names = _name_segments(profile)
    names[-1] = "Lapsed / Hibernating"
    f["segment"] = f["cluster"].map(names)
    f = f.join(_rfm_scores(f))

    total_rev = f["monetary"].sum()
    summary = (f.groupby("segment")
               .agg(customers=("frequency", "size"), revenue=("monetary", "sum"),
                    median_recency_days=("recency_days", "median"), median_orders=("frequency", "median"),
                    median_spend=("monetary", "median"), median_aov=("avg_order_value", "median"),
                    discount_share=("discount_share", "mean"), digital_share=("digital_share", "mean"),
                    median_tenure_days=("tenure_days", "median"), return_rate=("return_rate", "mean"),
                    n_categories=("n_categories", "mean"))
               .sort_values("revenue", ascending=False))
    summary["customer_share"] = summary["customers"] / summary["customers"].sum()
    summary["revenue_share"] = summary["revenue"] / total_rev
    summary["action"] = [PLAYBOOK.get(s.split(" ")[0] if s.startswith("Occasional") else s,
                                      PLAYBOOK["Occasional Shoppers"]) for s in summary.index]

    # validation against the simulator's hidden personas (not used for fitting)
    ari = None
    gt_path = cfg.raw_dir / "ground_truth_personas.csv"
    if gt_path.exists():
        gt = pd.read_csv(gt_path).set_index("customer_id")["persona"]
        common = f.index.intersection(gt.index)
        ari = float(adjusted_rand_score(gt.loc[common], f.loc[common, "segment"]))
        xt = pd.crosstab(f.loc[common, "segment"], gt.loc[common], normalize="index").round(3)
        save_table(xt.reset_index(), cfg.tables_dir / "segment_vs_true_persona.csv")

    _plot_segments(f[active], Z, summary.drop(index="Lapsed / Hibernating", errors="ignore"), summary, cfg)
    save_table(summary.round(4).reset_index(), cfg.tables_dir / "segment_summary.csv")
    save_table(f[["segment", "R", "F", "M", "RFM"] + PROFILE_FEATURES].round(3).reset_index(),
               cfg.tables_dir / "customer_segments.csv")
    f[["segment"]].to_pickle(cfg.processed_dir / "segments.pkl")

    out = dict(k=best_k, silhouette=scores, ari_vs_true_personas=ari,
               active_customers=int(active.sum()), lapsed_customers=int((~active).sum()),
               segments=summary.round(4).reset_index().to_dict("records"))
    save_json(out, cfg.reports_dir / "results" / "segmentation.json")
    return out


def _plot_segments(f, Z, active_summary, summary, cfg):
    order = list(summary.index)
    colors = {s: viz.PALETTE[i] for i, s in enumerate(active_summary.index)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.05, 1]))

    # left: customer share vs revenue share - the core story
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    h = 0.36
    ax.barh(y + h / 2, summary["customer_share"], height=h, color=viz.NEUTRAL, label="Share of customers")
    ax.barh(y - h / 2, summary["revenue_share"], height=h, color=viz.PALETTE[0], label="Share of revenue")
    for yi, (cs, rs) in zip(y, summary[["customer_share", "revenue_share"]].to_numpy()):
        ax.text(cs, yi + h / 2, f" {cs:.0%}", va="center", fontsize=8, color=viz.INK_2)
        ax.text(rs, yi - h / 2, f" {rs:.0%}", va="center", fontsize=8, color=viz.INK)
    ax.set_yticks(y, order)
    viz.pct_axis(ax, "x")
    ax.set_xlim(0, max(summary["customer_share"].max(), summary["revenue_share"].max()) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    viz.titled(ax, "Who drives revenue", "Customer share vs revenue share by behavioural segment")

    # right: PCA map of customers
    ax = axes[1]
    pcs = PCA(n_components=2, random_state=cfg.seed).fit_transform(Z)
    idx = np.random.default_rng(cfg.seed).choice(len(pcs), size=min(4000, len(pcs)), replace=False)
    for s in active_summary.index:
        m = (f["segment"].to_numpy()[idx] == s)
        ax.scatter(pcs[idx][m, 0], pcs[idx][m, 1], s=9, alpha=0.55, color=colors[s], label=s, linewidths=0)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), markerscale=2)
    viz.titled(ax, "Segment map of active customers (PCA)", "4,000-customer sample, behaviour features standardised")
    viz.save(fig, cfg.figures_dir / "10_customer_segments.png")
