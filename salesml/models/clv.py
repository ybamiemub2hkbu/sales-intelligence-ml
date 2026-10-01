"""Customer lifetime value: predicted net revenue over the next 180 days.

Compared against the finance team's usual heuristic (trailing-12-month
run-rate) on an out-of-time snapshot.  The CLV score is then crossed with the
churn score to build a value x risk action matrix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .. import viz
from ..config import Config
from ..features import customer_snapshot, future_outcomes, model_matrix
from ..utils import get_logger, rmse, save_json, save_table, top_decile_capture

log = get_logger(__name__)
ACTIVE_DAYS = 365
RISK_CUT = 0.30          # same boundary as the churn "Medium" tier


def _dataset(tx, customers, as_of, horizon):
    f = customer_snapshot(tx, as_of, customers)
    f = f[f["recency_days"] <= ACTIVE_DAYS]
    fut = future_outcomes(tx, as_of, horizon)
    return f.join(fut, how="left").fillna({"future_orders": 0, "future_revenue": 0})


def _models(seed, fast):
    signed_log = FunctionTransformer(lambda X: np.sign(X) * np.log1p(np.abs(X)))
    return {
        "Tweedie GLM": make_pipeline(signed_log, StandardScaler(),
                                     TweedieRegressor(power=1.5, link="log", alpha=0.05, max_iter=3000)),
        "Gradient boosting (Poisson)": HistGradientBoostingRegressor(
            loss="poisson", max_iter=200 if fast else 400, learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed),
    }


def _metrics(name, y, p):
    return dict(model=name, mae=mean_absolute_error(y, p), rmse=rmse(y, p),
                spearman=float(spearmanr(y, p).statistic), top_decile_capture=top_decile_capture(y, p),
                total_bias=float(p.sum() / y.sum() - 1))


def run(data: dict, cfg: Config) -> dict:
    tx, customers = data["transactions"], data["customers"]
    H = cfg.clv_horizon_days
    end = tx["order_date"].max() + pd.Timedelta(days=1)
    test_asof = end - pd.Timedelta(days=H)
    train_asofs = [test_asof - pd.Timedelta(days=H + k * 60) for k in range(3)]
    train = pd.concat([_dataset(tx, customers, s, H) for s in train_asofs])
    test = _dataset(tx, customers, test_asof, H)
    Xtr, ytr = model_matrix(train), train["future_revenue"].to_numpy()
    Xte, yte = model_matrix(test), test["future_revenue"].to_numpy()

    rows = [_metrics("Run-rate baseline (last 12m x 180/365)", yte, test["revenue_365d"].to_numpy() * H / 365)]
    preds, fitted = {}, {}
    for name, m in _models(cfg.seed, cfg.fast).items():
        m.fit(Xtr, ytr)
        preds[name] = np.clip(m.predict(Xte), 0, None)
        fitted[name] = m
        rows.append(_metrics(name, yte, preds[name]))
    board = pd.DataFrame(rows).sort_values("mae")
    best_name = board[board.model.isin(preds)].iloc[0]["model"]
    log.info("CLV leaderboard:\n%s", board.round(3).to_string(index=False))

    # decile calibration on test
    p_best = preds[best_name]
    dec = pd.qcut(pd.Series(p_best).rank(method="first"), 10, labels=False) + 1
    dec_tab = pd.DataFrame({"decile": dec, "predicted": p_best, "actual": yte}).groupby("decile").mean()

    # ---- score current customers
    full = pd.concat([train, test])
    model = fitted[best_name]
    model.fit(model_matrix(full), full["future_revenue"].to_numpy())
    now = customer_snapshot(tx, end, customers)
    now = now[now["recency_days"] <= ACTIVE_DAYS].copy()
    now["predicted_clv_180d"] = np.clip(model.predict(model_matrix(now)), 0, None)

    churn_path = cfg.processed_dir / "churn_scores.pkl"
    seg_path = cfg.processed_dir / "segments.pkl"
    if churn_path.exists():
        now = now.join(pd.read_pickle(churn_path), how="left")
    if seg_path.exists():
        now = now.join(pd.read_pickle(seg_path), how="left")
    quad = None
    if "churn_probability" in now:
        v_cut = now["predicted_clv_180d"].quantile(0.75)
        hi_v = now["predicted_clv_180d"] >= v_cut
        hi_r = now["churn_probability"] >= RISK_CUT
        now["value_risk_action"] = np.select(
            [hi_v & hi_r, hi_v & ~hi_r, ~hi_v & hi_r], ["Protect", "Grow & reward", "Automate win-back"],
            "Maintain")
        quad = (now.groupby("value_risk_action")
                .agg(customers=("predicted_clv_180d", "size"), predicted_clv_180d=("predicted_clv_180d", "sum"),
                     avg_churn_prob=("churn_probability", "mean"), revenue_365d=("revenue_365d", "sum"))
                .reindex(["Protect", "Grow & reward", "Automate win-back", "Maintain"]))
        quad["clv_share"] = quad["predicted_clv_180d"] / quad["predicted_clv_180d"].sum()
        save_table(quad.round(4).reset_index(), cfg.tables_dir / "value_risk_matrix.csv")
    _plot(dec_tab, now, best_name, board, cfg, v_cut if quad is not None else None)

    keep = ["predicted_clv_180d", "revenue_365d", "frequency", "recency_days"] + \
           [c for c in ("churn_probability", "segment", "value_risk_action") if c in now]
    save_table(now[keep].sort_values("predicted_clv_180d", ascending=False).round(4).reset_index(),
               cfg.tables_dir / "customer_clv_scores.csv")
    save_table(board.round(4), cfg.tables_dir / "clv_model_leaderboard.csv")

    base = board[board.model.str.startswith("Run-rate")].iloc[0]
    bst = board[board.model == best_name].iloc[0]
    out = dict(
        horizon_days=H, best_model=best_name, leaderboard=board.round(4).to_dict("records"),
        mae_improvement_vs_runrate=float(1 - bst["mae"] / base["mae"]),
        top_decile_capture=float(bst["top_decile_capture"]), spearman=float(bst["spearman"]),
        predicted_clv_total_next_180d=float(now["predicted_clv_180d"].sum()),
        top10pct_customers_clv_share=float(now["predicted_clv_180d"].nlargest(max(len(now) // 10, 1)).sum()
                                           / now["predicted_clv_180d"].sum()),
        value_risk=quad.round(4).reset_index().to_dict("records") if quad is not None else [],
    )
    save_json(out, cfg.reports_dir / "results" / "clv.json")
    return out


def _plot(dec_tab, now, best_name, board, cfg, v_cut):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    ax = axes[0]
    x = dec_tab.index.to_numpy()
    w = 0.38
    ax.bar(x - w / 2, dec_tab["actual"], width=w, color=viz.NEUTRAL, label="Actual (next 180 days)")
    ax.bar(x + w / 2, dec_tab["predicted"], width=w, color=viz.PALETTE[0], label=f"Predicted ({best_name})")
    ax.set_xticks(x, [f"D{i}" for i in x])
    ax.set_xlabel("Predicted-value decile (D10 = most valuable)")
    viz.money_axis(ax)
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    cap = board.set_index("model").loc[best_name, "top_decile_capture"]
    viz.titled(ax, "CLV model calibration by decile",
               f"Out-of-time test. Top 10% of predicted customers deliver {cap:.0%} of actual future revenue")

    ax = axes[1]
    if "value_risk_action" in now:
        colors = {"Protect": viz.PALETTE[1], "Grow & reward": viz.PALETTE[0],
                  "Automate win-back": viz.PALETTE[2], "Maintain": viz.NEUTRAL}
        s = now.sample(min(4000, len(now)), random_state=cfg.seed)
        for k in ["Maintain", "Automate win-back", "Grow & reward", "Protect"]:
            d = s[s["value_risk_action"] == k]
            ax.scatter(d["churn_probability"], d["predicted_clv_180d"] + 1, s=9, alpha=0.6, color=colors[k],
                       label=k, linewidths=0)
        ax.set_yscale("log")
        ax.axvline(RISK_CUT, color=viz.AXIS, lw=1)
        ax.axhline(v_cut + 1, color=viz.AXIS, lw=1)
        ax.set_xlabel("Churn probability (next 120 days)")
        ax.set_ylabel("Predicted CLV, next 180 days (log)")
        viz.money_axis(ax)
        viz.pct_axis(ax, "x")
        ax.legend(loc="lower left", markerscale=2.2, ncol=2)
        viz.titled(ax, "Value x risk action matrix", f"High value = top quartile of predicted CLV; elevated risk = churn probability >= {RISK_CUT:.0%}")
    viz.save(fig, cfg.figures_dir / "13_clv_value_risk.png")
