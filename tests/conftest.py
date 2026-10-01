import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salesml.config import Config  # noqa: E402
from salesml.data.cleaning import clean_transactions  # noqa: E402
from salesml.data.generator import SalesDataGenerator  # noqa: E402


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("salesml")
    return Config(n_customers=500, start_date="2023-01-01", end_date="2024-06-30",
                  data_dir=root / "data", reports_dir=root / "reports", models_dir=root / "models", fast=True)


@pytest.fixture(scope="session")
def raw(small_cfg):
    return SalesDataGenerator(small_cfg).generate()


@pytest.fixture(scope="session")
def clean_tx(raw):
    tx = raw["transactions"].copy()
    tx["order_date"] = tx["order_date"].astype("datetime64[ns]")
    out, _ = clean_transactions(tx, raw["products"])
    return out
