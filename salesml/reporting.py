"""Turn model outputs into decisions: recommendations + executive summary.

Every number in the executive summary is read from ``reports/results/*.json``
written by the analysis modules, so the report is regenerated - never
hand-edited - whenever the data or models change.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import Config
from .utils import get_logger, load_json, money, pct, save_json

log = get_logger(__name__)
FEATURE_LABELS = {
    "recency_days": "days since last order", "frequency": "number of orders", "orders_90d": "orders in the last 90 days",
    "avg_interpurchase_days": "usual gap between orders", "revenue_90d": "spend in the last 90 days",
    "recency_vs_cadence": "days since last order vs usual gap", "tenure_days": "customer tenure",
    "revenue_365d": "spend in the last 365 days", "order_trend": "order trend", "monetary": "lifetime spend",
}
MODULES = ["descriptive", "forecasting", "segmentation", "churn", "clv", "elasticity", "basket",
           "anomaly", "promotions"]


def load_results(cfg: Config) -> dict:
    out = {}
    for m in MODULES:
        p = cfg.reports_dir / "results" / f"{m}.json"
        if p.exists():
            out[m] = load_json(p)
    return out


def build_recommendations(r: dict) -> list[dict]:
    """Ranked, quantified actions. Each has an owner, an impact and the evidence."""
    recs = []
    if "promotions" in r:
        p = r["promotions"]
        bad = [s for s in p["summary"] if s["verdict"] == "Redesign / stop"]
        loss = -sum(s["total_incremental_gp"] for s in bad)
        worst = sorted(bad, key=lambda s: s["total_incremental_gp"])[:3]
        recs.append(dict(
            area="Promotions", owner="Marketing", impact_value=loss / 3,
            title="Cut discount depth on loss-making promotions",
            detail=(f"{len(bad)} of {len(p['summary'])} recurring promotions lose gross profit once the sales that "
                    f"would have happened anyway are removed (overall ROI {p['overall_roi']:+.2f}). Worst: "
                    + ", ".join(f"{w['promo_name']} ({money(w['total_incremental_gp'])})" for w in worst)
                    + f". Only {pct(p['incremental_share_of_promo_revenue'])} of revenue booked during promotions is "
                      "truly incremental. Test shallower discounts and target Deal Hunters only."),
            impact=f"~{money(loss / 3)} gross profit per year recovered"))
    if "elasticity" in r:
        e = r["elasticity"]
        top = sorted(e["by_category"], key=lambda c: -c["annual_profit_uplift"])[:3]
        recs.append(dict(
            area="Pricing", owner="Merchandising", impact_value=e["total_annual_profit_uplift"],
            title="Re-price within a +/-10% guardrail using measured elasticities",
            detail=(f"{e['n_raise']} products can take a price increase and {e['n_lower']} should come down. "
                    "Largest gains: " + ", ".join(f"{c['category']} ({money(c['annual_profit_uplift'])})" for c in top)
                    + ". Roll out as an A/B price test on 20% of stores first."),
            impact=f"+{money(e['total_annual_profit_uplift'])} gross profit per year "
                   f"(+{pct(e['total_annual_profit_uplift'] / e['current_annual_profit'])})"))
    if "churn" in r:
        c = r["churn"]
        recs.append(dict(
            area="Retention", owner="CRM", impact_value=c["recommended_contacts_profit_at_stake"]
            * 0.25,
            title=f"Run a model-targeted retention campaign to {c['recommended_contacts_now']:,} customers",
            detail=(f"{c['high_risk_customers_now']:,} active customers have >60% probability of not buying in the "
                    f"next {c['horizon_days']} days ({money(c['revenue_at_risk_now'])} revenue at risk). On the "
                    f"out-of-time test, contacting the top {pct(c['optimal_contact_share'], 0)} ranked by "
                    f"risk x value maximised profit; contacting everyone would lose money. Strongest signals: "
                    + ", ".join(FEATURE_LABELS.get(f, f.replace("_", " ")) for f in c["top_drivers"][:3]) + "."),
            impact=f"{money(c['recommended_contacts_profit_at_stake'])} gross profit at stake in the target list"))
    if "basket" in r:
        b = r["basket"]
        top = b["bundles"][:3]
        recs.append(dict(
            area="Merchandising", owner="E-commerce & Stores", impact_value=None,
            title="Launch bundles and 'frequently bought together' for the strongest product pairs",
            detail=("Top pairs: " + "; ".join(f"{x['bundle']} (lift {x['lift']:.0f}x, {pct(x['confidence'], 0)} attach)"
                                             for x in top)
                    + f". Orders containing these pairs average {money(top[0]['avg_order_value_with_bundle'])} vs "
                      f"{money(b['single_order_aov'])} overall. Co-locate them in store and on product pages."),
            impact="Higher basket size on the highest-affinity pairs"))
    if "forecasting" in r:
        f = r["forecasting"]
        cats = sorted(f["by_category"], key=lambda c: -c["expected_growth"])
        recs.append(dict(
            area="Planning", owner="Supply chain & Finance", impact_value=None,
            title=f"Plan next quarter at {money(f['next_quarter_total']['p50'])} "
                  f"({f['next_quarter_total']['growth']:+.1%} YoY)",
            detail=(f"80% interval {money(f['next_quarter_total']['p10'])} - {money(f['next_quarter_total']['p90'])}. "
                    f"Fastest growth expected in {cats[0]['category']} ({cats[0]['expected_growth']:+.0%}) and "
                    f"{cats[1]['category']} ({cats[1]['expected_growth']:+.0%}); slowest in {cats[-1]['category']} "
                    f"({cats[-1]['expected_growth']:+.0%}). Forecast error is "
                    f"{pct(f['error_reduction_vs_baseline'], 0)} lower than the seasonal-naive method."),
            impact="Inventory sized to demand, fewer stock-outs and markdowns"))
    if "segmentation" in r:
        s = {x["segment"]: x for x in r["segmentation"]["segments"]}
        if "Champions" in s and "New & Promising" in s:
            ch, nw = s["Champions"], s["New & Promising"]
            recs.append(dict(
                area="Customers", owner="CRM", impact_value=None,
                title="Protect Champions; move new customers to a second purchase fast",
                detail=(f"Champions are {pct(ch['customer_share'], 0)} of customers but {pct(ch['revenue_share'], 0)} "
                        f"of revenue. {nw['customers']:,} New & Promising customers have bought once; a 30-day "
                        "onboarding journey is the cheapest lever to convert them into regulars."),
                impact="Revenue concentration de-risked, higher repeat rate"))
    if "anomaly" in r:
        a = r["anomaly"]
        recs.append(dict(
            area="Operations", owner="Ops & IT", impact_value=-a["lost_revenue_from_drops"] / 3,
            title="Put automated incident alerts on daily sales",
            detail=(f"{a['incidents']} material incidents were detected across {a['series_monitored']} monitored "
                    f"series, costing {money(-a['lost_revenue_from_drops'])} in unexplained drops, including a "
                    "website outage and a 3-week stock-out. Alerting the same day shortens time-to-fix."),
            impact=f"Up to {money(-a['lost_revenue_from_drops'] / 3)} per year of revenue protected"))
    if "descriptive" in r:
        d = r["descriptive"]
        h = d["return_hotspot"]
        recs.append(dict(
            area="Returns", owner="Product & E-commerce", impact_value=None,
            title=f"Fix online returns in {h['category']}",
            detail=(f"Online {h['category']} return rate is {pct(h['rate'])} vs {pct(h['instore_rate'])} in store; "
                    f"returns erased {money(d['kpis']['revenue_lost_to_returns'])} of revenue overall. Better size "
                    "guides, fit reviews and photos target the gap."),
            impact="Lower reverse-logistics cost, higher net revenue"))
    ranked = sorted([x for x in recs if x["impact_value"]], key=lambda x: -x["impact_value"])
    return ranked + [x for x in recs if not x["impact_value"]]


def write_executive_summary(cfg: Config, r: dict, recs: list[dict]) -> str:
    d, k = r["descriptive"], r["descriptive"]["kpis"]
    f, s, c, v = r["forecasting"], r["segmentation"], r["churn"], r["clv"]
    e, b, a, p = r["elasticity"], r["basket"], r["anomaly"], r["promotions"]
    yrs = sorted(k["revenue_by_year"])
    L = []
    w = L.append
    w(f"# Executive Summary - {cfg.company_name}")
    w("")
    w(f"*Auto-generated by `python main.py` on {date.today().isoformat()} from "
      f"{k['orders']:,} orders, {k['period_start'][:10]} to {k['period_end'][:10]}. "
      "Company and data are simulated (see README).*")
    w("")
    w("## The business at a glance")
    w("")
    w("| Metric | Value |")
    w("|---|---|")
    w(f"| Net revenue (3 years) | **{money(k['net_revenue'])}** |")
    w(f"| Net revenue {yrs[-1]} | {money(k['revenue_by_year'][yrs[-1]])} ({k['yoy_growth_last_year']:+.1%} YoY) |")
    w(f"| Gross margin | {pct(k['gross_margin'])} |")
    w(f"| Orders / customers | {k['orders']:,} / {k['customers']:,} |")
    w(f"| Average order value | {money(k['avg_order_value'], 2)} |")
    w(f"| Active customers (last 365 days) | {k['active_customers_365d']:,} |")
    w(f"| Revenue lost to returns | {money(k['revenue_lost_to_returns'])} ({pct(k['return_rate_revenue'])}) |")
    w(f"| Top 20% of customers' share of revenue | {pct(d['concentration']['top20_share'], 0)} |")
    w("")
    w("![Monthly revenue](figures/01_monthly_revenue_yoy.png)")
    w("")
    w("## Recommended actions (ranked by quantified impact)")
    w("")
    for i, x in enumerate(recs, 1):
        w(f"### {i}. {x['title']}")
        w(f"**Owner:** {x['owner']} · **Impact:** {x['impact']}")
        w("")
        w(x["detail"])
        w("")
    w("## What the models found")
    w("")
    w("### Demand forecast - next quarter")
    w(f"* Production method: **{f['production_method']}**, chosen by {f['n_origins']} rolling-origin backtests "
      f"(category-week WAPE {pct(f['production_wape'])} vs {pct(f['baseline_wape'])} seasonal-naive).")
    w(f"* Next 13 weeks: **{money(f['next_quarter_total']['p50'])}** "
      f"(80% interval {money(f['next_quarter_total']['p10'])}-{money(f['next_quarter_total']['p90'])}), "
      f"{f['next_quarter_total']['growth']:+.1%} vs the same weeks last year.")
    w(f"* {len(f['repaired_weeks'])} incident weeks (e.g. the February stock-out) were repaired before training "
      "so they are not mistaken for seasonality.")
    w("")
    w("![Forecast](figures/09_forecast_next_quarter.png)")
    w("")
    w("### Customer segments")
    w("| Segment | Customers | Revenue share | Median orders | Action |")
    w("|---|---|---|---|---|")
    for x in s["segments"]:
        w(f"| {x['segment']} | {x['customers']:,} | {pct(x['revenue_share'], 0)} | {x['median_orders']:.0f} | {x['action']} |")
    w("")
    w("![Segments](figures/10_customer_segments.png)")
    w("")
    w("### Churn and lifetime value")
    w(f"* Churn model (**{c['best_model']}**): out-of-time ROC-AUC **{c['test_roc_auc']:.3f}**, "
      f"top-decile lift **{c['top_decile_lift']:.2f}x**; base churn rate {pct(c['base_churn_rate_test'])}.")
    w(f"* CLV model (**{v['best_model']}**): MAE {pct(v['mae_improvement_vs_runrate'], 0)} lower than the run-rate "
      f"heuristic; the top 10% of predicted customers deliver {pct(v['top_decile_capture'], 0)} of next-180-day revenue.")
    vr = {x["value_risk_action"]: x for x in v["value_risk"]}
    if vr:
        w(f"* Value x risk: **{vr['Protect']['customers']:,}** high-value customers at elevated risk (personal outreach), "
          f"{vr['Grow & reward']['customers']:,} high-value / low-risk customers hold "
          f"{pct(vr['Grow & reward']['clv_share'], 0)} of predicted value.")
    w("")
    w("![Churn](figures/11_churn_model_evaluation.png)")
    w("")
    w("### Pricing")
    w("| Category | Elasticity (95% CI) | Profit uplift / yr |")
    w("|---|---|---|")
    bc = {x["category"]: x for x in e["by_category"]}
    for x in e["elasticities"]:
        w(f"| {x['category']} | {x['elasticity']:.2f} ({x['ci_low']:.2f} to {x['ci_high']:.2f}) | "
          f"{money(bc[x['category']]['annual_profit_uplift'])} |")
    w("")
    w("![Elasticity](figures/14_price_elasticity.png)")
    w("")
    w("### Basket, promotions, incidents")
    w(f"* {pct(b['multi_item_order_share'], 0)} of orders contain 2+ products; {b['n_strong_rules']} strong rules (lift >= 2).")
    w(f"* Promotions: {money(p['total_discount_cost'])} of discounts generated {money(p['total_incremental_gp'])} "
      f"incremental gross profit (ROI {p['overall_roi']:+.2f}). Best: {p['best']}. Worst: {p['worst']}.")
    w(f"* Anomaly detection: {a['incidents']} material incidents across {a['series_monitored']} series.")
    w("")
    w("![Promotions](figures/17_promotion_roi.png)")
    w("")
    w("![Anomalies](figures/16_anomaly_detection.png)")
    w("")
    w("## Validation against simulation ground truth")
    w("Because the data is simulated, we know the true mechanisms and can check whether the models recover them:")
    w("")
    w(f"* **Price elasticity:** mean absolute error {e['mean_abs_error_vs_truth']:.2f}; the 95% CI covers the true value "
      f"in {pct(e['truth_coverage'], 0)} of categories.")
    if b.get("complements_recovered"):
        w(f"* **Market basket:** {b['complements_recovered']['found']} of {b['complements_recovered']['total']} planted "
          "product affinities appear among the top 40 rules.")
    w(f"* **Anomalies:** {a['detected_injected']} of {a['injected_total']} planted incidents raised an alert:")
    for x in a["validation"]:
        w(f"  * {x['incident']} ({x['window']}): {'detected' if x['detected'] else 'not alerted'} - strongest signal "
          f"{x['strongest_series']} at |z| = {x['strongest_abs_z']:.1f}")
    w("")
    w("*See `reports/tables/` for every number above and `docs/METHODOLOGY.md` for the methods.*")
    text = "\n".join(L) + "\n"
    (cfg.reports_dir / "EXECUTIVE_SUMMARY.md").write_text(text, encoding="utf-8")
    return text


def run(cfg: Config) -> dict:
    r = load_results(cfg)
    missing = [m for m in MODULES if m not in r]
    if missing:
        raise RuntimeError(f"run these steps first: {missing}")
    recs = build_recommendations(r)
    save_json(recs, cfg.reports_dir / "results" / "recommendations.json")
    write_executive_summary(cfg, r, recs)
    log.info("executive summary written with %d recommendations", len(recs))
    return dict(recommendations=recs)
