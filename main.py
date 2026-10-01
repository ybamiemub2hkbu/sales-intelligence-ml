"""Sales Intelligence pipeline - one command from raw data to decisions.

    python main.py                       # everything
    python main.py --steps churn clv     # selected steps (reuses processed data)
    python main.py --fast                # quicker models, for a smoke test
    python main.py --n-customers 5000    # smaller simulated company
    python main.py --fast --out /tmp/run # write data/reports somewhere else (used by CI)
"""
from __future__ import annotations

import argparse
import time

from salesml.config import Config
from salesml.utils import get_logger

log = get_logger("pipeline")

STEPS = ["generate", "clean", "descriptive", "forecast", "segment", "churn", "clv",
         "elasticity", "basket", "anomaly", "promotions", "report", "dashboard"]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS + ["all"], default=["all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-customers", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="smaller models / fewer bootstrap draws")
    ap.add_argument("--out", default=None, help="base folder for data/, reports/ and models/ (default: project root)")
    args = ap.parse_args(argv)

    cfg = Config(seed=args.seed, fast=args.fast)
    if args.n_customers:
        cfg.n_customers = args.n_customers
    if args.out:
        from pathlib import Path
        base = Path(args.out)
        cfg.data_dir, cfg.reports_dir, cfg.models_dir = base / "data", base / "reports", base / "models"
    cfg.ensure_dirs()
    steps = STEPS if "all" in args.steps else [s for s in STEPS if s in args.steps]

    from salesml.data import cleaning, generator
    data = None

    def processed():
        nonlocal data
        if data is None:
            data = cleaning.load_processed(cfg)
        return data

    t_all = time.perf_counter()
    for step in steps:
        t0 = time.perf_counter()
        log.info("=" * 18 + f" {step.upper()} " + "=" * 18)
        if step == "generate":
            generator.generate_and_save(cfg)
            data = None
        elif step == "clean":
            data = cleaning.build_processed(cfg)
        elif step == "descriptive":
            from salesml.analysis import descriptive
            descriptive.run(processed(), cfg)
        elif step == "forecast":
            from salesml.models import forecasting
            forecasting.run(processed(), cfg)
        elif step == "segment":
            from salesml.models import segmentation
            segmentation.run(processed(), cfg)
        elif step == "churn":
            from salesml.models import churn
            churn.run(processed(), cfg)
        elif step == "clv":
            from salesml.models import clv
            clv.run(processed(), cfg)
        elif step == "elasticity":
            from salesml.models import elasticity
            elasticity.run(processed(), cfg)
        elif step == "basket":
            from salesml.models import basket
            basket.run(processed(), cfg)
        elif step == "anomaly":
            from salesml.models import anomaly
            anomaly.run(processed(), cfg)
        elif step == "promotions":
            from salesml.models import promotions
            promotions.run(processed(), cfg)
        elif step == "report":
            from salesml import reporting
            reporting.run(cfg)
        elif step == "dashboard":
            from salesml import dashboard
            dashboard.build(processed(), cfg)
        log.info("%s finished in %.1fs", step, time.perf_counter() - t0)
    log.info("pipeline finished in %.1fs -> see reports/EXECUTIVE_SUMMARY.md and reports/dashboard/index.html",
             time.perf_counter() - t_all)


if __name__ == "__main__":
    main()
