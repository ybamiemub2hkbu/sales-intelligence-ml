"""Build the interactive dashboard: one self-contained HTML file.

All model outputs are embedded as JSON, so the page needs no server and can be
opened from disk or hosted on GitHub Pages (``docs/index.html``).  Charts are
drawn client-side with Chart.js; simulators (retention budget, price changes)
recompute in the browser from the embedded model outputs.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, Config
from .models.forecasting import weekly_series
from .utils import get_logger, load_json

log = get_logger(__name__)
TEMPLATE = Path(__file__).parent / "templates" / "dashboard.html"


def _records(df: pd.DataFrame, decimals: int = 4) -> list[dict]:
    return json.loads(df.round(decimals).to_json(orient="records", date_format="iso"))


def build_payload(data: dict, cfg: Config) -> dict:
    tx = data["transactions"]
    res = cfg.reports_dir / "results"
    tab = cfg.tables_dir
    R = {p.stem: load_json(p) for p in res.glob("*.json")}
    d = R["descriptive"]

    monthly = tx.groupby("month")["net_revenue"].sum()
    q = tx.assign(q=tx["order_date"].dt.to_period("Q").astype(str))
    mix = q.pivot_table(index="q", columns="channel", values="net_revenue", aggfunc="sum")
    mix = mix.div(mix.sum(1), axis=0)[["In-Store", "Online", "Mobile App"]]

    # forecast: last 52 weeks + next quarter
    ws = weekly_series(tx).iloc[-52:]
    fut = pd.read_csv(tab / "forecast_next_quarter_weekly.csv")
    hist = {c: [[w.strftime("%Y-%m-%d"), round(float(v), 2)] for w, v in ws[c].items()] for c in ws.columns}

    # customers: CLV + churn + segment for every active customer
    clv = pd.read_csv(tab / "customer_clv_scores.csv")
    risk = pd.read_csv(tab / "churn_risk_scores.csv")[["customer_id", "expected_profit_loss", "risk_tier",
                                                        "order_trend", "target_for_retention"]]
    cust = clv.merge(risk, on="customer_id", how="left")
    cust_cols = ["customer_id", "segment", "churn_probability", "predicted_clv_180d", "revenue_365d",
                 "frequency", "recency_days", "value_risk_action", "expected_profit_loss", "target_for_retention"]
    cust = cust[cust_cols]
    cust_rows = [[r.customer_id, r.segment, round(r.churn_probability, 4), round(r.predicted_clv_180d, 2),
                  round(r.revenue_365d, 2), int(r.frequency), int(r.recency_days), r.value_risk_action,
                  round(r.expected_profit_loss, 2), int(r.target_for_retention)] for r in cust.itertuples()]

    sim = pd.read_pickle(cfg.processed_dir / "churn_test_predictions.pkl")
    sim_rows = [[round(a, 4), round(b, 2), int(c)] for a, b, c in sim.itertuples(index=False)]

    prices = pd.read_csv(tab / "price_recommendations.csv")
    lift = pd.read_csv(tab / "basket_rules_categories.csv")
    promo_plan = pd.read_csv(tab / "promotion_plan_next_year_review.csv")
    daily = pd.read_csv(tab / "anomaly_daily_total.csv")
    incidents = pd.read_csv(tab / "anomaly_incidents.csv") if (tab / "anomaly_incidents.csv").exists() else pd.DataFrame()

    return dict(
        meta=dict(company=cfg.company_name, start=d["kpis"]["period_start"][:10], end=d["kpis"]["period_end"][:10],
                  generated=date.today().isoformat(), orders=d["kpis"]["orders"]),
        kpis=d["kpis"], concentration=d["concentration"],
        monthly=[[m.strftime("%Y-%m"), round(float(v), 2)] for m, v in monthly.items()],
        category=_records(pd.DataFrame(d["category"])[["category", "net", "gp", "margin", "disc"]]),
        channel_mix=dict(quarters=list(mix.index), series={c: mix[c].round(4).tolist() for c in mix.columns}),
        returns=d["return_hotspot"],
        recommendations=R["recommendations"],
        forecast=dict(history=hist, future=_records(fut), by_category=R["forecasting"]["by_category"],
                      metrics=R["forecasting"]["metrics"], production_method=R["forecasting"]["production_method"],
                      total=R["forecasting"]["next_quarter_total"], repaired=R["forecasting"]["repaired_weeks"]),
        segments=R["segmentation"]["segments"],
        churn={k: R["churn"][k] for k in ("leaderboard", "best_model", "test_roc_auc", "top_decile_lift",
                                          "base_churn_rate_test", "horizon_days", "high_risk_customers_now",
                                          "revenue_at_risk_now", "active_customers_now", "by_segment",
                                          "recommended_contacts_now")},
        churn_importance=_records(pd.read_csv(tab / "churn_feature_importance.csv").head(10)),
        retention_sim=dict(rows=sim_rows, default_cost=cfg.retention_offer_cost,
                           default_success=cfg.retention_success_rate),
        clv={k: R["clv"][k] for k in ("leaderboard", "best_model", "value_risk", "mae_improvement_vs_runrate",
                                      "top_decile_capture")},
        customers=dict(columns=cust_cols, rows=cust_rows),
        elasticity=R["elasticity"]["elasticities"],
        pricing=dict(products=_records(prices), guardrail=cfg.price_change_guardrail,
                     total_uplift=R["elasticity"]["total_annual_profit_uplift"],
                     by_category=R["elasticity"]["by_category"]),
        basket=dict(bundles=R["basket"]["bundles"], multi_share=R["basket"]["multi_item_order_share"],
                    aov=R["basket"]["single_order_aov"],
                    lift=_records(lift[["antecedent", "consequent", "lift", "support", "confidence"]])),
        promotions=dict(summary=R["promotions"]["summary"], overall_roi=R["promotions"]["overall_roi"],
                        discount=R["promotions"]["total_discount_cost"],
                        incremental_gp=R["promotions"]["total_incremental_gp"],
                        incremental_share=R["promotions"]["incremental_share_of_promo_revenue"],
                        plan=_records(promo_plan.assign(start_date=promo_plan["start_date"].astype(str).str[:10],
                                                        end_date=promo_plan["end_date"].astype(str).str[:10]))),
        anomaly=dict(daily=daily[["date", "actual", "expected"]].values.tolist(),
                     incidents=_records(incidents) if len(incidents) else [],
                     validation=R["anomaly"]["validation"], series=R["anomaly"]["series_monitored"],
                     threshold=cfg.anomaly_z_threshold),
        quality=dict(report=_records(pd.read_csv(tab / "data_quality_report.csv")),
                     elasticity_mae=R["elasticity"]["mean_abs_error_vs_truth"],
                     elasticity_coverage=R["elasticity"]["truth_coverage"],
                     basket_recovered=R["basket"]["complements_recovered"],
                     anomaly_detected=[R["anomaly"]["detected_injected"], R["anomaly"]["injected_total"]]),
    )


def render(payload: dict, standalone: bool = True) -> str:
    body = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(payload, separators=(",", ":"), default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    body = body.replace("/*__PAYLOAD__*/null", blob.replace("</", "<\\/"))
    if not standalone:
        return body
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            "</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def build(data: dict, cfg: Config) -> Path:
    payload = build_payload(data, cfg)
    html = render(payload)
    out = cfg.reports_dir / "dashboard" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    if cfg.reports_dir.resolve() == (ROOT / "reports").resolve():   # GitHub Pages copy for the main run only
        pages = ROOT / "docs" / "index.html"
        pages.parent.mkdir(parents=True, exist_ok=True)
        pages.write_text(html, encoding="utf-8")
    log.info("dashboard written to %s (%.0f KB)", out, len(html) / 1024)
    return out
