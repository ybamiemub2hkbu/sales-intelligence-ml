"""Market basket analysis: which products sell together?

Pairwise association rules (support, confidence, lift) computed directly with
pandas - for retail catalogues of a few hundred SKUs this is exact, fast and
needs no extra dependency.  Rules feed bundle design, cross-sell widgets and
store layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import viz
from ..config import Config
from ..utils import get_logger, load_json, save_json, save_table

log = get_logger(__name__)


def pair_rules(baskets: pd.DataFrame, item_col: str, min_pair_orders: int) -> pd.DataFrame:
    """Association rules A -> B for every item pair seen in >= min_pair_orders orders."""
    b = baskets[["order_id", item_col]].drop_duplicates()
    n_orders = b["order_id"].nunique()
    item_cnt = b[item_col].value_counts()
    multi = b[b["order_id"].map(b["order_id"].value_counts()) > 1]
    pairs = multi.merge(multi, on="order_id", suffixes=("_a", "_b"))
    pairs = pairs[pairs[f"{item_col}_a"] != pairs[f"{item_col}_b"]]
    cnt = pairs.groupby([f"{item_col}_a", f"{item_col}_b"]).size().rename("pair_orders").reset_index()
    cnt = cnt[cnt["pair_orders"] >= min_pair_orders]
    cnt["support"] = cnt["pair_orders"] / n_orders
    cnt["confidence"] = cnt["pair_orders"] / cnt[f"{item_col}_a"].map(item_cnt)
    cnt["consequent_support"] = cnt[f"{item_col}_b"].map(item_cnt) / n_orders
    cnt["lift"] = cnt["confidence"] / cnt["consequent_support"]
    return cnt.rename(columns={f"{item_col}_a": "antecedent", f"{item_col}_b": "consequent"}) \
              .sort_values("lift", ascending=False).reset_index(drop=True)


def run(data: dict, cfg: Config) -> dict:
    tx, products = data["transactions"], data["products"]
    names = products.set_index("product_id")["product_name"]
    cats = products.set_index("product_id")["category"]
    n_orders = tx["order_id"].nunique()
    multi_share = float((tx.groupby("order_id")["product_id"].nunique() > 1).mean())

    rules = pair_rules(tx, "product_id", cfg.basket_min_pair_orders)
    rules["antecedent_name"] = rules["antecedent"].map(names)
    rules["consequent_name"] = rules["consequent"].map(names)
    rules["antecedent_category"] = rules["antecedent"].map(cats)
    rules["consequent_category"] = rules["consequent"].map(cats)
    strong = rules[(rules["lift"] >= 2) & (rules["confidence"] >= 0.10)]

    # value of a bundle: average revenue of orders containing the pair
    order_rev = tx.groupby("order_id")["net_revenue"].sum()
    single_aov = float(order_rev.mean())

    # category-level lift matrix
    crules = pair_rules(tx, "category", cfg.basket_min_pair_orders)
    cat_order = viz.CATEGORIES
    lift_mx = crules.pivot(index="antecedent", columns="consequent", values="lift").reindex(
        index=cat_order, columns=cat_order)

    # validation: how many simulated complement pairs are in the top rules
    recovered = None
    gt_path = cfg.raw_dir / "ground_truth.json"
    if gt_path.exists():
        comp = load_json(gt_path)["complements"]
        top = strong.head(40)
        pairs = set(zip(top["antecedent_name"], top["consequent_name"]))
        hits = sum(((c["a"], c["b"]) in pairs) or ((c["b"], c["a"]) in pairs) for c in comp)
        recovered = dict(found=int(hits), total=len(comp))

    # unique (unordered) pairs for the bundle list
    seen, bundles = set(), []
    for r in strong.itertuples(index=False):
        key = frozenset((r.antecedent, r.consequent))
        if key in seen:
            continue
        seen.add(key)
        both = set(tx.loc[tx["product_id"] == r.antecedent, "order_id"]) & \
            set(tx.loc[tx["product_id"] == r.consequent, "order_id"])
        bundles.append(dict(bundle=f"{r.antecedent_name} + {r.consequent_name}",
                            categories=f"{r.antecedent_category} / {r.consequent_category}",
                            orders_together=int(r.pair_orders), lift=float(r.lift),
                            confidence=float(r.confidence),
                            avg_order_value_with_bundle=float(order_rev.loc[list(both)].mean())))
        if len(bundles) >= 15:
            break
    bundles = pd.DataFrame(bundles)

    _plot(strong, lift_mx, cfg)
    save_table(rules.round(5), cfg.tables_dir / "basket_rules_products.csv")
    save_table(crules.round(5), cfg.tables_dir / "basket_rules_categories.csv")
    save_table(bundles.round(3), cfg.tables_dir / "bundle_recommendations.csv")

    off_diag = crules[crules["antecedent"] != crules["consequent"]].sort_values("lift", ascending=False)
    out = dict(orders=n_orders, multi_item_order_share=multi_share, n_rules=len(rules), n_strong_rules=len(strong),
               single_order_aov=single_aov, bundles=bundles.round(3).to_dict("records"),
               complements_recovered=recovered,
               top_category_pairs=off_diag.head(6)[["antecedent", "consequent", "lift"]].round(3).to_dict("records"))
    save_json(out, cfg.reports_dir / "results" / "basket.json")
    log.info("%d strong rules; ground-truth complements recovered: %s", len(strong), recovered)
    return out


def _plot(strong, lift_mx, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axes[0]
    seen, rows = set(), []
    for r in strong.itertuples(index=False):
        k = frozenset((r.antecedent, r.consequent))
        if k not in seen:
            seen.add(k)
            rows.append(r)
        if len(rows) == 12:
            break
    top = pd.DataFrame(rows)[::-1]
    labels = [f"{a}  +  {b}" for a, b in zip(top["antecedent_name"], top["consequent_name"])]
    ax.barh(labels, top["lift"], color=viz.PALETTE[0], height=0.62)
    for i, (l, c) in enumerate(zip(top["lift"], top["confidence"])):
        ax.text(l, i, f"  {l:.1f}x  ({c:.0%} attach)", va="center", fontsize=8.3, color=viz.INK_2)
    ax.set_xlim(0, top["lift"].max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Lift (how much more often bought together than by chance)")
    viz.titled(ax, "Strongest product affinities", "Top 12 unique pairs; attach = P(second item | first item)")

    ax = axes[1]
    m = np.log2(lift_mx.to_numpy(dtype=float))
    lim = np.nanmax(np.abs(m))
    im = ax.imshow(m, cmap=viz.DIV_CMAP, vmin=-lim, vmax=lim)
    short = [c.split(" ")[0].replace("&", "") for c in lift_mx.index]
    ax.set_xticks(range(len(short)), short, rotation=35, ha="right")
    ax.set_yticks(range(len(short)), lift_mx.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i != j and not np.isnan(m[i, j]):
                ax.text(j, i, f"{lift_mx.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if abs(m[i, j]) > 0.6 * lim else viz.INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("log2 lift (0 = independent)", color=viz.INK_2, fontsize=8.5)
    viz.titled(ax, "Category cross-shopping", "Red = bought together more than chance, blue = less")
    viz.save(fig, cfg.figures_dir / "15_market_basket.png")
