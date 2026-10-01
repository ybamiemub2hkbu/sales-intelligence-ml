"""Customer churn prediction and retention targeting.

Definition: an *active* customer (ordered in the last 365 days) churns if they
place no order in the next ``churn_horizon_days``.

Validation is **out-of-time**: models train on stacked historical snapshots
and are tested on a later snapshot whose outcome window never overlaps
training labels - the same way the model will be used in production.

The output is not just a probability: it is a ranked contact list and the
contact depth that maximises expected retained profit after offer costs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .. import viz
from ..config import Config
from ..features import customer_snapshot, future_outcomes, model_matrix
from ..utils import get_logger, save_json, save_table

log = get_logger(__name__)
ACTIVE_DAYS = 365


def build_dataset(tx, customers, as_of, horizon):
    f = customer_snapshot(tx, as_of, customers)
    f = f[f["recency_days"] <= ACTIVE_DAYS]
    fut = future_outcomes(tx, as_of, horizon)
    f = f.join(fut, how="left").fillna({"future_orders": 0, "future_revenue": 0})
    f["churned"] = (f["future_orders"] == 0).astype(int)
    return f


def candidate_models(seed: int, fast: bool):
    signed_log = FunctionTransformer(lambda X: np.sign(X) * np.log1p(np.abs(X)))
    return {
        "Logistic regression": make_pipeline(signed_log, StandardScaler(),
                                             LogisticRegression(max_iter=2000, C=0.5)),
        "Random forest": RandomForestClassifier(n_estimators=150 if fast else 400, min_samples_leaf=20,
                                                max_features=0.4, n_jobs=-1, random_state=seed),
        "Gradient boosting": HistGradientBoostingClassifier(max_iter=200 if fast else 400, learning_rate=0.04,
                                                            max_leaf_nodes=31, min_samples_leaf=40,
                                                            l2_regularization=1.0, random_state=seed),
    }


def retention_profit_curve(p_churn, churned, value, cost, success):
    """Expected net profit when contacting the top-k customers ranked by p * value."""
    order = np.argsort(-(p_churn * value))
    saved = churned[order] * success * value[order]
    net = np.cumsum(saved) - cost * np.arange(1, len(order) + 1)
    share = np.arange(1, len(order) + 1) / len(order)
    return share, net, order


def run(data: dict, cfg: Config) -> dict:
    tx, customers = data["transactions"], data["customers"]
    H = cfg.churn_horizon_days
    end = tx["order_date"].max() + pd.Timedelta(days=1)
    test_asof = end - pd.Timedelta(days=H)
    train_asofs = [test_asof - pd.Timedelta(days=H + k * cfg.churn_snapshot_step_days)
                   for k in range(cfg.churn_n_train_snapshots)]

    train = pd.concat([build_dataset(tx, customers, s, H) for s in train_asofs])
    test = build_dataset(tx, customers, test_asof, H)
    Xtr, ytr = model_matrix(train), train["churned"].to_numpy()
    Xte, yte = model_matrix(test), test["churned"].to_numpy()
    groups = train.index.to_numpy()
    log.info("churn train rows %d (snapshots %s), test rows %d, base churn rate %.1f%%",
             len(train), [s.date().isoformat() for s in train_asofs], len(test), 100 * yte.mean())

    # ---- model selection with customer-grouped CV
    cv_rows, test_rows, fitted, test_pred = [], [], {}, {}
    for name, model in candidate_models(cfg.seed, cfg.fast).items():
        aucs, aps = [], []
        for tr, va in GroupKFold(n_splits=4).split(Xtr, ytr, groups):
            m = model.__class__(**model.get_params()) if not hasattr(model, "steps") else \
                make_pipeline(*[s for _, s in model.steps])
            m.fit(Xtr.iloc[tr], ytr[tr])
            p = m.predict_proba(Xtr.iloc[va])[:, 1]
            aucs.append(roc_auc_score(ytr[va], p))
            aps.append(average_precision_score(ytr[va], p))
        model.fit(Xtr, ytr)
        fitted[name] = model
        p = model.predict_proba(Xte)[:, 1]
        test_pred[name] = p
        cv_rows.append(dict(model=name, cv_roc_auc=np.mean(aucs), cv_roc_auc_std=np.std(aucs),
                            cv_pr_auc=np.mean(aps)))
        test_rows.append(dict(model=name, test_roc_auc=roc_auc_score(yte, p),
                              test_pr_auc=average_precision_score(yte, p),
                              test_brier=brier_score_loss(yte, p)))
        log.info("%-20s CV AUC %.3f | out-of-time AUC %.3f", name, np.mean(aucs), test_rows[-1]["test_roc_auc"])
    leaderboard = pd.DataFrame(cv_rows).merge(pd.DataFrame(test_rows), on="model").sort_values(
        "cv_roc_auc", ascending=False)
    best_name = leaderboard.iloc[0]["model"]
    best = fitted[best_name]
    p_test = test_pred[best_name]

    # ---- lift and targeting economics on the out-of-time test snapshot
    margin = float(tx["gross_profit"].sum() / tx["net_revenue"].sum())
    value = (test["revenue_365d"].to_numpy() * H / 365) * margin  # profit at stake over horizon
    share, net, _ = retention_profit_curve(p_test, yte, value, cfg.retention_offer_cost, cfg.retention_success_rate)
    k_best = int(np.argmax(net))
    rng = np.random.default_rng(cfg.seed)
    rand_net = np.cumsum(yte[rng.permutation(len(yte))] * cfg.retention_success_rate *
                         value[rng.permutation(len(yte))]) - cfg.retention_offer_cost * np.arange(1, len(yte) + 1)
    # decile 1 = the 10% of customers with the highest predicted risk
    decile = 10 - pd.qcut(pd.Series(p_test).rank(method="first"), 10, labels=False)
    lift = pd.Series(yte).groupby(decile.to_numpy()).mean().sort_index() / yte.mean()

    imp = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=5 if cfg.fast else 8,
                                 random_state=cfg.seed, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False)

    # ---- score today's active customers with a model refit on all labelled data
    full = pd.concat([train, test])
    best.fit(model_matrix(full), full["churned"].to_numpy())
    now = customer_snapshot(tx, end, customers)
    now = now[now["recency_days"] <= ACTIVE_DAYS].copy()
    now["churn_probability"] = best.predict_proba(model_matrix(now))[:, 1]
    now["profit_at_stake"] = now["revenue_365d"] * H / 365 * margin
    now["expected_profit_loss"] = now["churn_probability"] * now["profit_at_stake"]
    now["revenue_at_risk"] = now["churn_probability"] * now["revenue_365d"] * H / 365
    seg_path = cfg.processed_dir / "segments.pkl"
    if seg_path.exists():
        now = now.join(pd.read_pickle(seg_path), how="left")
    now["risk_tier"] = pd.cut(now["churn_probability"], [0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"],
                              include_lowest=True)
    contact_n = int(round((k_best + 1) / len(yte) * len(now)))
    now["target_for_retention"] = 0
    now.loc[now["expected_profit_loss"].nlargest(contact_n).index, "target_for_retention"] = 1
    risk_list = now.sort_values("expected_profit_loss", ascending=False)

    # out-of-time predictions power the dashboard's retention-budget simulator
    pd.DataFrame({"p_churn": p_test, "profit_at_stake": value, "churned": yte}).to_pickle(
        cfg.processed_dir / "churn_test_predictions.pkl")
    _plot(test_pred, yte, leaderboard, share, net, rand_net, k_best, importance, p_test, best_name, lift, cfg)
    save_table(leaderboard.round(4), cfg.tables_dir / "churn_model_leaderboard.csv")
    save_table(importance.rename("auc_drop").round(5).reset_index().rename(columns={"index": "feature"}),
               cfg.tables_dir / "churn_feature_importance.csv")
    cols = ["churn_probability", "risk_tier", "expected_profit_loss", "revenue_at_risk", "revenue_365d",
            "recency_days", "frequency", "order_trend", "target_for_retention"] + \
           (["segment"] if "segment" in risk_list else [])
    save_table(risk_list[cols].round(4).reset_index(), cfg.tables_dir / "churn_risk_scores.csv")
    risk_list[["churn_probability"]].to_pickle(cfg.processed_dir / "churn_scores.pkl")

    by_seg = (risk_list.groupby("segment").agg(customers=("churn_probability", "size"),
                                               avg_churn_prob=("churn_probability", "mean"),
                                               revenue_at_risk=("revenue_at_risk", "sum"))
              .sort_values("revenue_at_risk", ascending=False) if "segment" in risk_list else pd.DataFrame())
    if len(by_seg):
        save_table(by_seg.round(4).reset_index(), cfg.tables_dir / "churn_by_segment.csv")

    out = dict(
        horizon_days=H, test_snapshot=test_asof, train_snapshots=train_asofs,
        base_churn_rate_test=float(yte.mean()), best_model=best_name,
        leaderboard=leaderboard.round(4).to_dict("records"),
        test_roc_auc=float(leaderboard.set_index("model").loc[best_name, "test_roc_auc"]),
        test_pr_auc=float(leaderboard.set_index("model").loc[best_name, "test_pr_auc"]),
        top_decile_lift=float(lift.iloc[0]),
        top_drivers=list(importance.index[:6]),
        optimal_contact_share=float(share[k_best]), optimal_net_profit_test=float(net[k_best]),
        contact_all_net_profit_test=float(net[-1]),
        active_customers_now=int(len(now)),
        high_risk_customers_now=int((now["risk_tier"] == "High").sum()),
        revenue_at_risk_now=float(now["revenue_at_risk"].sum()),
        high_risk_revenue_at_risk=float(now.loc[now["risk_tier"] == "High", "revenue_at_risk"].sum()),
        recommended_contacts_now=contact_n,
        recommended_contacts_profit_at_stake=float(now.loc[now["target_for_retention"] == 1, "expected_profit_loss"].sum()),
        by_segment=by_seg.round(4).reset_index().to_dict("records") if len(by_seg) else [],
    )
    save_json(out, cfg.reports_dir / "results" / "churn.json")
    return out


def _plot(test_pred, yte, leaderboard, share, net, rand_net, k_best, importance, p_best, best_name, lift, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    ax = axes[0]
    for i, (name, p) in enumerate(test_pred.items()):
        fpr, tpr, _ = roc_curve(yte, p)
        ax.plot(fpr, tpr, color=viz.PALETTE[i], lw=2, label=f"{name}  AUC {roc_auc_score(yte, p):.3f}")
    ax.plot([0, 1], [0, 1], color=viz.NEUTRAL, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    viz.titled(ax, "Out-of-time ROC", "Test snapshot never seen in training")

    ax = axes[1]
    frac, mean_p = calibration_curve(yte, p_best, n_bins=10, strategy="quantile")
    ax.plot([0, 1], [0, 1], color=viz.NEUTRAL, lw=1)
    ax.plot(mean_p, frac, color=viz.PALETTE[0], marker="o", ms=5, mec=viz.SURFACE, mew=1.5)
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Observed churn rate")
    viz.pct_axis(ax)
    viz.pct_axis(ax, "x")
    viz.titled(ax, f"Calibration ({best_name})", f"Top-decile lift: {lift.iloc[0]:.2f}x the base churn rate")

    ax = axes[2]
    ax.plot(share, net, color=viz.PALETTE[0], lw=2, label="Ranked by model (p x value)")
    ax.plot(share, rand_net, color=viz.NEUTRAL, lw=1.6, label="Random targeting")
    ax.axhline(0, color=viz.AXIS, lw=1)
    ax.scatter([share[k_best]], [net[k_best]], s=50, color=viz.PALETTE[0], edgecolor=viz.SURFACE, linewidth=2, zorder=5)
    ax.annotate(f"Contact top {share[k_best]:.0%}\nnet ${net[k_best]:,.0f}", (share[k_best], net[k_best]),
                xytext=(12, -6), textcoords="offset points", fontsize=9, va="top")
    viz.pct_axis(ax, "x")
    viz.money_axis(ax)
    ax.set_xlabel("Share of active customers contacted")
    ax.legend(loc="lower left")
    viz.titled(ax, "Retention campaign profit curve", "Saved margin minus offer cost, test snapshot")
    viz.save(fig, cfg.figures_dir / "11_churn_model_evaluation.png")

    top = importance.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(top.index, top.values, color=viz.PALETTE[0], height=0.62)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Drop in ROC-AUC when feature is shuffled")
    viz.titled(ax, "What predicts churn", "Permutation importance on the out-of-time test set")
    viz.save(fig, cfg.figures_dir / "12_churn_drivers.png")
