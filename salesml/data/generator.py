"""Realistic synthetic sales data for a multi-channel retailer.

Why simulate?  Real transactional data is confidential, and public retail
datasets rarely contain prices, promotions, returns, channels *and* customer
histories together.  The simulator below encodes the mechanisms that real
sales data exhibits so that every downstream model has something genuine to
discover - and, because we know the truth, we can *verify* that the models
recover it (see ``ground_truth.json``).

Mechanisms encoded
------------------
* Customer personas (loyalist, regular, deal seeker, occasional, one-and-done)
  with different purchase rates, basket sizes, promo sensitivity and lifetimes.
* Engagement decay before a customer churns (the signal churn models learn).
* Growing customer acquisition + a shift from in-store to mobile over time.
* Weekly, yearly, holiday and category-specific seasonality.
* A promotion calendar (category and site-wide events, incl. 2026 planned)
  that lifts traffic, shifts category mix and pulls demand forward.
* Product-level list-price changes and constant-elasticity demand response.
* Complementary products that are bought together (market-basket signal).
* Returns that depend on category, channel and discount depth.
* Operational incidents: a website outage, a viral product spike, a supply
  stock-out and a regional storm.
* Dirty records: duplicates, missing channels, negative quantities and
  prices keyed in cents - just like a real export.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..utils import get_logger, save_json

log = get_logger(__name__)

# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------
# season = relative demand by calendar month (Jan..Dec)
CATEGORY_SPECS: dict[str, dict] = {
    "Electronics": dict(
        elasticity=-2.2, margin=0.24, return_rate=0.07, share=0.17, qty_lambda=0.10,
        season=[0.85, 0.80, 0.85, 0.88, 0.92, 0.95, 1.00, 1.08, 1.00, 1.00, 1.30, 1.55],
        products=[("Wireless Earbuds", 79), ("Bluetooth Speaker", 59), ("Smartwatch", 199),
                  ("Phone Case", 25), ("USB-C Fast Charger", 29), ("Power Bank 20K", 45),
                  ("10in Tablet", 329), ("Noise-Cancelling Headphones", 249),
                  ("Smart Plug 2-Pack", 32), ("Action Camera", 289), ("E-Reader", 139),
                  ("Gaming Mouse", 59), ("Tablet Sleeve", 29)]),
    "Home & Kitchen": dict(
        elasticity=-1.5, margin=0.42, return_rate=0.05, share=0.16, qty_lambda=0.25,
        season=[0.95, 0.90, 0.95, 1.00, 1.05, 0.95, 0.90, 0.95, 1.00, 1.05, 1.15, 1.30],
        products=[("Chef Knife Set", 119), ("Cast Iron Skillet", 49), ("Burr Coffee Grinder", 89),
                  ("French Press", 35), ("Non-Stick Pan Set", 129), ("Cutting Board", 29),
                  ("Stand Mixer", 279), ("Air Fryer", 119), ("Linen Towel Set", 39),
                  ("Glass Storage Set", 45), ("Electric Kettle", 49), ("Dinnerware 16pc", 89),
                  ("Scented Candle", 22)]),
    "Outdoor & Camping": dict(
        elasticity=-1.3, margin=0.38, return_rate=0.06, share=0.12, qty_lambda=0.15,
        season=[0.45, 0.50, 0.75, 1.05, 1.45, 1.75, 1.75, 1.45, 1.00, 0.70, 0.55, 0.65],
        products=[("2-Person Tent", 189), ("Sleeping Bag", 99), ("Camping Stove", 69),
                  ("Headlamp", 29), ("Hiking Backpack 40L", 139), ("Water Filter", 39),
                  ("Camp Chair", 49), ("Trekking Poles", 59), ("Cooler 30L", 159),
                  ("Insect Repellent", 12), ("Sleeping Pad", 79), ("Camp Lantern", 35)]),
    "Apparel": dict(
        elasticity=-1.8, margin=0.55, return_rate=0.16, share=0.17, qty_lambda=0.30,
        season=[0.80, 0.80, 1.10, 1.15, 1.00, 0.90, 0.85, 1.00, 1.15, 1.15, 1.10, 1.15],
        products=[("Rain Jacket", 129), ("Fleece Pullover", 69), ("Merino Base Layer", 79),
                  ("Hiking Boots", 159), ("Trail Running Shoes", 129), ("Wool Socks 3-Pack", 24),
                  ("Denim Jacket", 89), ("Organic Cotton Tee", 25), ("Chino Pants", 59),
                  ("Down Vest", 119), ("Sun Hat", 29), ("Leather Belt", 39)]),
    "Sports & Fitness": dict(
        elasticity=-1.6, margin=0.40, return_rate=0.06, share=0.11, qty_lambda=0.20,
        season=[1.55, 1.25, 1.05, 0.95, 1.05, 1.00, 0.90, 0.85, 0.90, 0.90, 0.95, 1.10],
        products=[("Yoga Mat", 39), ("Adjustable Dumbbells", 249), ("Resistance Bands", 25),
                  ("Insulated Water Bottle", 29), ("Foam Roller", 29), ("Jump Rope", 15),
                  ("Fitness Tracker", 99), ("Kettlebell 12kg", 59), ("Gym Bag", 49),
                  ("Protein Shaker", 12), ("Yoga Block Set", 19)]),
    "Beauty & Care": dict(
        elasticity=-1.1, margin=0.60, return_rate=0.03, share=0.11, qty_lambda=0.50,
        season=[0.95, 1.20, 1.00, 1.00, 1.15, 1.00, 0.95, 0.95, 0.95, 0.95, 1.05, 1.25],
        products=[("Vitamin C Serum", 34), ("Daily Moisturizer", 28), ("Mineral Sunscreen", 19),
                  ("Shampoo Bar", 14), ("Beard Oil", 22), ("Lip Balm 4-Pack", 9),
                  ("Bamboo Toothbrush 4-Pack", 12), ("Body Wash", 11), ("Face Mask Set", 24),
                  ("Perfume 50ml", 69)]),
    "Toys & Games": dict(
        elasticity=-1.9, margin=0.45, return_rate=0.04, share=0.08, qty_lambda=0.25,
        season=[0.55, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.85, 1.00, 1.80, 3.10],
        products=[("Building Blocks 500pc", 49), ("Strategy Board Game", 39), ("Jigsaw Puzzle 1000pc", 22),
                  ("RC Car", 69), ("Plush Bear", 19), ("Science Kit", 35), ("Card Game", 15),
                  ("Wooden Train Set", 59), ("Art Supplies Kit", 29)]),
    "Gourmet Grocery": dict(
        elasticity=-0.8, margin=0.30, return_rate=0.01, share=0.08, qty_lambda=0.90,
        season=[0.95, 0.95, 0.95, 0.95, 1.00, 0.95, 0.95, 0.95, 1.00, 1.05, 1.15, 1.45],
        products=[("Single-Origin Coffee Beans", 18), ("Artisan Dark Chocolate", 8),
                  ("Extra Virgin Olive Oil", 22), ("Trail Mix 1kg", 16), ("Loose Leaf Tea", 14),
                  ("Raw Honey", 12), ("Hot Sauce Trio", 19), ("Gourmet Gift Basket", 79),
                  ("Dried Mango", 9)]),
}

# Pairs of products bought together (name_a, name_b, P(add b | a in basket))
COMPLEMENTS = [
    ("2-Person Tent", "Sleeping Bag", 0.45),
    ("Sleeping Bag", "Sleeping Pad", 0.35),
    ("Camping Stove", "Camp Lantern", 0.25),
    ("Hiking Boots", "Wool Socks 3-Pack", 0.45),
    ("Hiking Backpack 40L", "Water Filter", 0.30),
    ("Trekking Poles", "Hiking Boots", 0.20),
    ("10in Tablet", "Tablet Sleeve", 0.40),
    ("Smartwatch", "USB-C Fast Charger", 0.25),
    ("Wireless Earbuds", "Phone Case", 0.20),
    ("Burr Coffee Grinder", "Single-Origin Coffee Beans", 0.55),
    ("French Press", "Single-Origin Coffee Beans", 0.40),
    ("Yoga Mat", "Yoga Block Set", 0.45),
    ("Adjustable Dumbbells", "Gym Bag", 0.20),
    ("Fitness Tracker", "Insulated Water Bottle", 0.25),
    ("Mineral Sunscreen", "Sun Hat", 0.25),
    ("Stand Mixer", "Glass Storage Set", 0.20),
]


@dataclass(frozen=True)
class Persona:
    name: str
    share: float
    orders_per_year: float
    mean_lifetime_days: float
    promo_sensitivity: float
    extra_items: float          # Poisson mean of items beyond the first


PERSONAS = [
    Persona("loyalist", 0.12, 13.0, 2000, 0.4, 1.40),
    Persona("regular", 0.30, 5.5, 950, 1.0, 1.00),
    Persona("deal_seeker", 0.23, 3.2, 650, 2.8, 1.20),
    Persona("occasional", 0.25, 2.0, 480, 1.2, 0.70),
    Persona("one_and_done", 0.10, 1.0, 30, 1.0, 0.55),
]

REGIONS = {"North": 0.27, "South": 0.24, "East": 0.31, "West": 0.18}
REGION_CATEGORY_TILT = {  # regional taste differences
    "West": {"Outdoor & Camping": 1.45, "Sports & Fitness": 1.15},
    "North": {"Apparel": 1.20, "Home & Kitchen": 1.10},
    "South": {"Beauty & Care": 1.20, "Outdoor & Camping": 0.85},
    "East": {"Electronics": 1.15, "Gourmet Grocery": 1.20},
}
CHANNELS = ["In-Store", "Online", "Mobile App"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_SHARES = [0.13, 0.27, 0.25, 0.19, 0.16]
AGE_CATEGORY_TILT = {
    "18-24": {"Electronics": 1.4, "Apparel": 1.3, "Home & Kitchen": 0.6, "Gourmet Grocery": 0.7},
    "25-34": {"Electronics": 1.2, "Sports & Fitness": 1.3},
    "35-44": {"Toys & Games": 1.7, "Home & Kitchen": 1.2},
    "45-54": {"Home & Kitchen": 1.3, "Outdoor & Camping": 1.2},
    "55+": {"Gourmet Grocery": 1.6, "Home & Kitchen": 1.3, "Electronics": 0.6},
}

INCIDENTS = [
    dict(kind="website_outage", start="2024-03-12", end="2024-03-13",
         scope="channel", target=["Online", "Mobile App"], effect=-0.85),
    dict(kind="viral_social_post", start="2024-09-18", end="2024-09-21",
         scope="product", target=["Wireless Earbuds"], effect=7.0),
    dict(kind="supply_stockout", start="2025-02-03", end="2025-02-23",
         scope="category", target=["Outdoor & Camping"], effect=-0.80),
    dict(kind="regional_storm", start="2025-01-15", end="2025-01-17",
         scope="region_channel", target=["North", "In-Store"], effect=-0.90),
]


# --------------------------------------------------------------------------
# Calendar helpers
# --------------------------------------------------------------------------
def thanksgiving(year: int) -> pd.Timestamp:
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    return first_thu + pd.Timedelta(weeks=3)


def promotion_calendar(years: list[int]) -> pd.DataFrame:
    """Recurring promotional events. Discount is off list price."""
    rows = []
    for y in years:
        tg = thanksgiving(y)
        events = [
            ("New Year Fitness Push", f"{y}-01-02", f"{y}-01-15", 0.20, ["Sports & Fitness"]),
            ("Winter Apparel Clearance", f"{y}-01-20", f"{y}-01-31", 0.40, ["Apparel"]),
            ("Valentine's Beauty", f"{y}-02-07", f"{y}-02-14", 0.15, ["Beauty & Care"]),
            ("Spring Fashion Event", f"{y}-03-15", f"{y}-03-28", 0.25, ["Apparel"]),
            ("Mother's Day Gifts", f"{y}-05-01", f"{y}-05-11", 0.15, ["Beauty & Care", "Home & Kitchen"]),
            ("Summer Outdoor Sale", f"{y}-06-15", f"{y}-07-05", 0.20, ["Outdoor & Camping"]),
            ("Mid-Year Mega Sale", f"{y}-07-15", f"{y}-07-18", 0.15, ["ALL"]),
            ("Back to School Tech", f"{y}-08-08", f"{y}-08-25", 0.15, ["Electronics", "Apparel"]),
            ("Electronics Flash Weekend", f"{y}-10-11", f"{y}-10-13", 0.35, ["Electronics"]),
            ("Black Friday Week", (tg - pd.Timedelta(days=3)).strftime("%Y-%m-%d"),
             (tg + pd.Timedelta(days=4)).strftime("%Y-%m-%d"), 0.30, ["ALL"]),
            ("Holiday Toy Event", f"{y}-12-01", f"{y}-12-20", 0.20, ["Toys & Games"]),
            ("Gourmet Gifting", f"{y}-12-05", f"{y}-12-23", 0.10, ["Gourmet Grocery"]),
        ]
        for name, s, e, d, cats in events:
            rows.append(dict(promo_name=name, start_date=pd.Timestamp(s), end_date=pd.Timestamp(e),
                             discount_pct=d, categories="|".join(cats)))
    promos = pd.DataFrame(rows).sort_values("start_date").reset_index(drop=True)
    promos.insert(0, "promo_id", [f"PR{i:03d}" for i in range(1, len(promos) + 1)])
    return promos


def _periodic_interp(month_values: list[float], doy: np.ndarray) -> np.ndarray:
    """Smoothly interpolate 12 monthly multipliers to day-of-year."""
    centers = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    vals = np.asarray(month_values, float)
    x = np.concatenate([centers - 365, centers, centers + 365])
    v = np.concatenate([vals, vals, vals])
    return np.interp(doy, x, v)


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
class SalesDataGenerator:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
        self.n_days = len(self.dates)
        self.categories = list(CATEGORY_SPECS)
        self.cat_index = {c: i for i, c in enumerate(self.categories)}

    # ---------------------------------------------------------- products --
    def _build_products(self) -> pd.DataFrame:
        rows, pid = [], 1
        for cat, spec in CATEGORY_SPECS.items():
            for name, price in spec["products"]:
                elasticity = spec["elasticity"] + self.rng.normal(0, 0.12)
                margin = np.clip(spec["margin"] + self.rng.normal(0, 0.04), 0.12, 0.75)
                rows.append(dict(
                    product_id=f"P{pid:03d}", product_name=name, category=cat,
                    base_price=float(price), unit_cost=round(price * (1 - margin), 2),
                    popularity=float(self.rng.lognormal(0, 0.55)),
                    true_elasticity=float(elasticity)))
                pid += 1
        return pd.DataFrame(rows)

    def _price_paths(self, products: pd.DataFrame) -> np.ndarray:
        """List price per (day, product): piecewise-constant with drift."""
        n_p = len(products)
        prices = np.empty((self.n_days, n_p))
        for j, base in enumerate(products["base_price"].to_numpy()):
            level, day = 1.0, 0
            while day < self.n_days:
                span = int(self.rng.integers(45, 150))
                infl = 1 + 0.03 * day / 365  # mild inflation
                p = base * level * infl
                p = max(round(p) - 0.01, 0.99) if p > 5 else round(p, 2)
                prices[day:day + span, j] = p
                level = float(np.clip(level * self.rng.uniform(0.86, 1.14), 0.75, 1.30))
                day += span
        return prices

    # --------------------------------------------------------- customers --
    def _build_customers(self) -> pd.DataFrame:
        n = self.cfg.n_customers
        rng = self.rng
        start = self.dates[0]
        p_shares = np.array([p.share for p in PERSONAS])
        persona_idx = rng.choice(len(PERSONAS), size=n, p=p_shares / p_shares.sum())

        # 45% are pre-existing customers acquired in the three prior years
        legacy = rng.random(n) < 0.45
        legacy_days = rng.integers(1, 1095, size=n)
        # new customers: acquisition grows over time with Nov/Jan bumps
        doy = self.dates.dayofyear.to_numpy()
        acq_w = (np.linspace(1.0, 1.35, self.n_days)
                 * _periodic_interp([1.3, 1.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.05, 1.35, 1.45], doy))
        acq_day = rng.choice(self.n_days, size=n, p=acq_w / acq_w.sum())
        signup = np.where(legacy, (start - pd.to_timedelta(legacy_days, "D")).values,
                          self.dates[acq_day].values)
        signup = pd.to_datetime(signup)

        # mobile adoption rises with signup date
        frac = np.clip((signup - pd.Timestamp("2021-01-01")).days / (365 * 5), 0, 1)
        p_mobile = 0.10 + 0.35 * frac
        p_online = np.full(n, 0.33)
        p_store = 1 - p_mobile - p_online
        u = rng.random(n)
        channel = np.where(u < p_store, "In-Store", np.where(u < p_store + p_online, "Online", "Mobile App"))

        lifetimes = np.array([rng.exponential(PERSONAS[i].mean_lifetime_days) for i in persona_idx])
        lifetimes = np.maximum(lifetimes, 20)
        rate_mult = rng.gamma(4.0, 0.25, size=n)

        customers = pd.DataFrame(dict(
            customer_id=[f"C{i:05d}" for i in range(1, n + 1)],
            signup_date=signup.normalize(),
            region=rng.choice(list(REGIONS), size=n, p=list(REGIONS.values())),
            age_band=rng.choice(AGE_BANDS, size=n, p=AGE_SHARES),
            preferred_channel=channel,
            loyalty_member=rng.random(n) < np.where(persona_idx <= 1, 0.65, 0.20),
        ))
        customers["_persona"] = [PERSONAS[i].name for i in persona_idx]
        customers["_persona_idx"] = persona_idx
        customers["_lifetime_days"] = lifetimes
        customers["_rate_mult"] = rate_mult
        customers["_churn_date"] = customers["signup_date"] + pd.to_timedelta(lifetimes.round(), "D")
        return customers

    def _category_affinity(self, customers: pd.DataFrame) -> np.ndarray:
        base = np.array([CATEGORY_SPECS[c]["share"] for c in self.categories])
        aff = np.empty((len(customers), len(self.categories)))
        for i, age in enumerate(customers["age_band"].to_numpy()):
            tilt = np.array([AGE_CATEGORY_TILT.get(age, {}).get(c, 1.0) for c in self.categories])
            alpha = base * tilt
            aff[i] = self.rng.dirichlet(alpha / alpha.sum() * 6)
        return aff

    # ----------------------------------------------------- intensities --
    def _daily_components(self, promos: pd.DataFrame):
        d = self.dates
        doy = d.dayofyear.to_numpy()
        weekly = np.array([0.92, 0.90, 0.93, 0.97, 1.08, 1.18, 1.02])[d.dayofweek]
        yearly = _periodic_interp([0.90, 0.85, 0.95, 0.97, 1.00, 0.98, 1.00, 1.02, 0.95, 0.98, 1.10, 1.30], doy)
        holiday = np.ones(self.n_days)
        for y in sorted(set(d.year)):
            tg = thanksgiving(y)
            for day, mult in [(tg, 0.6), (pd.Timestamp(f"{y}-12-25"), 0.25),
                              (pd.Timestamp(f"{y}-12-24"), 0.8), (pd.Timestamp(f"{y}-01-01"), 0.55),
                              (tg + pd.Timedelta(days=1), 1.6), (tg + pd.Timedelta(days=4), 1.3)]:
                holiday[d == day] *= mult
            pre_xmas = (d >= f"{y}-12-08") & (d <= f"{y}-12-22")
            holiday[pre_xmas] *= 1.25
        base = weekly * yearly * holiday

        # promo traffic pressure and per-category discount on each day
        n_c = len(self.categories)
        disc = np.zeros((self.n_days, n_c))
        promo_ref = np.full((self.n_days, n_c), "", dtype=object)
        traffic = np.zeros(self.n_days)
        post_dip = np.ones(self.n_days)
        for _, p in promos.iterrows():
            mask = (d >= p.start_date) & (d <= p.end_date)
            if not mask.any() and p.end_date < d[0]:
                continue
            cats = self.categories if p.categories == "ALL" else p.categories.split("|")
            breadth = 1.0 if p.categories == "ALL" else 0.30 * len(cats)
            traffic[mask] += p.discount_pct * breadth * 1.2
            for c in cats:
                j = self.cat_index[c]
                better = mask & (p.discount_pct > disc[:, j])
                disc[better, j] = p.discount_pct
                promo_ref[better, j] = p.promo_id
            if p.categories == "ALL":  # demand pulled forward -> post-event dip
                after = (d > p.end_date) & (d <= p.end_date + pd.Timedelta(days=10))
                post_dip[after] *= 0.85
        cat_season = np.stack([_periodic_interp(CATEGORY_SPECS[c]["season"], doy)
                               for c in self.categories], axis=1)
        return base, traffic, post_dip, disc, promo_ref, cat_season

    # ------------------------------------------------------------ orders --
    def _simulate_orders(self, customers, base, traffic, post_dip) -> pd.DataFrame:
        rng = self.rng
        start, end = self.dates[0], self.dates[-1]
        intensity = {pi: base * (1 + p.promo_sensitivity * traffic) * post_dip
                     for pi, p in enumerate(PERSONAS)}
        o_cust, o_day = [], []
        cols = zip(customers["customer_id"], customers["signup_date"], customers["_churn_date"],
                   customers["_persona_idx"], customers["_rate_mult"])
        for cid, signup, churn, pi, rate_mult in cols:
            first = max(signup, start)
            last = min(churn, end)
            if last < first:
                continue
            a, b = (first - start).days, (last - start).days
            persona = PERSONAS[pi]
            w = intensity[pi][a:b + 1].copy()
            if churn <= end:  # engagement fades before churn
                k = min(120, len(w))
                w[-k:] *= np.linspace(1.0, 0.3, k)
            if persona.name == "one_and_done":
                n = 1 + int(rng.random() < 0.12)
            else:
                lam = persona.orders_per_year * rate_mult * w.sum() / 365
                n = rng.poisson(lam)
                if signup >= start and n == 0:
                    n = 1  # a signup implies a first purchase
            if n == 0:
                continue
            days = rng.choice(np.arange(a, b + 1), size=n, p=w / w.sum())
            if signup >= start:  # first order on signup day
                days[np.argmin(days)] = a
            o_cust.extend([cid] * n)
            o_day.extend(days.tolist())
        orders = pd.DataFrame(dict(customer_id=o_cust, day_idx=np.array(o_day, dtype=int)))
        orders = orders.sort_values(["day_idx", "customer_id"]).reset_index(drop=True)
        return orders

    # ------------------------------------------------------------- build --
    def generate(self) -> dict[str, pd.DataFrame]:
        rng = self.rng
        years = sorted(set(self.dates.year)) + [self.dates[-1].year + 1]
        promos = promotion_calendar(years)
        products = self._build_products()
        prices = self._price_paths(products)
        customers = self._build_customers()
        affinity = self._category_affinity(customers)
        base, traffic, post_dip, disc, promo_ref, cat_season = self._daily_components(promos)

        log.info("simulating orders for %d customers", len(customers))
        orders = self._simulate_orders(customers, base, traffic, post_dip)

        cust = customers.set_index("customer_id")
        cpos = pd.Series(np.arange(len(customers)), index=customers["customer_id"])
        orders["cpos"] = cpos.loc[orders["customer_id"]].to_numpy()
        orders["region"] = cust["region"].to_numpy()[orders["cpos"]]
        pref = cust["preferred_channel"].to_numpy()[orders["cpos"]]
        mobile_trend = 0.15 + 0.25 * orders["day_idx"].to_numpy() / self.n_days
        rand_channel = np.where(rng.random(len(orders)) < mobile_trend, "Mobile App",
                                np.where(rng.random(len(orders)) < 0.55, "Online", "In-Store"))
        orders["channel"] = np.where(rng.random(len(orders)) < 0.75, pref, rand_channel)

        # incidents that remove orders
        drop = np.zeros(len(orders), bool)
        for inc in INCIDENTS:
            in_window = ((orders["day_idx"] >= (pd.Timestamp(inc["start"]) - self.dates[0]).days)
                         & (orders["day_idx"] <= (pd.Timestamp(inc["end"]) - self.dates[0]).days))
            if inc["scope"] == "channel":
                hit = in_window & orders["channel"].isin(inc["target"])
            elif inc["scope"] == "region_channel":
                hit = in_window & (orders["region"] == inc["target"][0]) & (orders["channel"] == inc["target"][1])
            else:
                continue
            drop |= hit.to_numpy() & (rng.random(len(orders)) < -inc["effect"])
        orders = orders[~drop].reset_index(drop=True)
        orders["order_id"] = [f"O{i:07d}" for i in range(1, len(orders) + 1)]
        log.info("orders: %d", len(orders))

        # -------- lines: number of items per order
        persona_idx = cust["_persona_idx"].to_numpy()[orders["cpos"]]
        extra = np.array([p.extra_items for p in PERSONAS])[persona_idx]
        extra = extra * np.where(orders["channel"] == "Mobile App", 0.85, 1.0)
        n_items = 1 + rng.poisson(extra)
        lines = orders.loc[orders.index.repeat(n_items)].reset_index(drop=True)
        day = lines["day_idx"].to_numpy()
        lp_idx = persona_idx.repeat(n_items)
        sens = np.array([p.promo_sensitivity for p in PERSONAS])[lp_idx]

        # -------- category choice
        n_c = len(self.categories)
        region_tilt = np.ones((len(REGIONS), n_c))
        reg_list = list(REGIONS)
        for r, tilts in REGION_CATEGORY_TILT.items():
            for c, m in tilts.items():
                region_tilt[reg_list.index(r), self.cat_index[c]] = m
        reg_idx = pd.Categorical(lines["region"], categories=reg_list).codes
        P = (affinity[lines["cpos"].to_numpy()] * cat_season[day] * region_tilt[reg_idx]
             * np.power(1 + 2.0 * disc[day], sens[:, None]))
        stock_cat = next(i for i in INCIDENTS if i["kind"] == "supply_stockout")
        s0 = (pd.Timestamp(stock_cat["start"]) - self.dates[0]).days
        s1 = (pd.Timestamp(stock_cat["end"]) - self.dates[0]).days
        P[(day >= s0) & (day <= s1), self.cat_index[stock_cat["target"][0]]] *= (1 + stock_cat["effect"])
        P /= P.sum(1, keepdims=True)
        cat_choice = (P.cumsum(1) < rng.random(len(lines))[:, None]).sum(1)
        cat_choice = np.minimum(cat_choice, n_c - 1)

        # -------- product choice within category (price-elastic)
        viral = next(i for i in INCIDENTS if i["kind"] == "viral_social_post")
        v0 = (pd.Timestamp(viral["start"]) - self.dates[0]).days
        v1 = (pd.Timestamp(viral["end"]) - self.dates[0]).days
        prod_choice = np.empty(len(lines), int)
        for j, cat in enumerate(self.categories):
            sel = np.where(cat_choice == j)[0]
            pidx = products.index[products["category"] == cat].to_numpy()
            pr = products.loc[pidx]
            W = (pr["popularity"].to_numpy()[None, :]
                 * np.power(prices[:, pidx] / pr["base_price"].to_numpy()[None, :],
                            pr["true_elasticity"].to_numpy()[None, :]))
            vir = np.where(pr["product_name"].isin(viral["target"]))[0]
            if len(vir):
                W[v0:v1 + 1, vir] *= viral["effect"]
            Wd = W[day[sel]]
            Wd /= Wd.sum(1, keepdims=True)
            k = (Wd.cumsum(1) < rng.random(len(sel))[:, None]).sum(1)
            prod_choice[sel] = pidx[np.minimum(k, len(pidx) - 1)]
        lines["pidx"] = prod_choice

        # -------- complements: add the companion product to the basket
        name_to_idx = dict(zip(products["product_name"], products.index))
        comp_rows = []
        for a, b, p in COMPLEMENTS:
            hit = np.where((lines["pidx"].to_numpy() == name_to_idx[a]) & (rng.random(len(lines)) < p))[0]
            if len(hit):
                extra_rows = lines.iloc[hit].copy()
                extra_rows["pidx"] = name_to_idx[b]
                comp_rows.append(extra_rows)
        if comp_rows:
            lines = pd.concat([lines, *comp_rows], ignore_index=True)

        # -------- quantity, collapse duplicates within an order
        cat_of = products["category"].to_numpy()[lines["pidx"]]
        qlam = np.array([CATEGORY_SPECS[c]["qty_lambda"] for c in cat_of])
        lines["quantity"] = 1 + rng.poisson(qlam)
        lines = (lines.groupby(["order_id", "pidx"], as_index=False, sort=False)
                 .agg(quantity=("quantity", "sum"), day_idx=("day_idx", "first"),
                      customer_id=("customer_id", "first"), region=("region", "first"),
                      channel=("channel", "first")))
        lines = lines.sort_values(["day_idx", "order_id"]).reset_index(drop=True)

        # -------- prices, discounts, returns
        day = lines["day_idx"].to_numpy()
        pidx = lines["pidx"].to_numpy()
        cidx = np.array([self.cat_index[c] for c in products["category"].to_numpy()[pidx]])
        list_price = prices[day, pidx]
        promo_disc = disc[day, cidx]
        coupon = np.where(rng.random(len(lines)) < 0.04, 0.10, 0.0)
        discount = np.maximum(promo_disc, coupon)
        promo_id = np.where(promo_disc >= coupon, promo_ref[day, cidx], "")
        promo_id = np.where(discount > 0, promo_id, "")
        unit_price = np.round(list_price * (1 - discount), 2)
        qty = lines["quantity"].to_numpy()
        rr = np.array([CATEGORY_SPECS[c]["return_rate"] for c in products["category"].to_numpy()[pidx]])
        rr = rr * np.where(discount >= 0.25, 1.4, 1.0) * np.where(lines["channel"] != "In-Store", 1.35, 1.0)

        tx = pd.DataFrame(dict(
            order_id=lines["order_id"],
            order_date=self.dates[day].strftime("%Y-%m-%d"),
            customer_id=lines["customer_id"],
            product_id=products["product_id"].to_numpy()[pidx],
            category=products["category"].to_numpy()[pidx],
            channel=lines["channel"], region=lines["region"],
            quantity=qty, list_price=list_price, discount_pct=discount,
            unit_price=unit_price, revenue=np.round(unit_price * qty, 2),
            unit_cost=products["unit_cost"].to_numpy()[pidx],
            promo_id=promo_id,
            is_returned=(rng.random(len(lines)) < rr).astype(int),
        ))
        tx.insert(1, "line_number", tx.groupby("order_id").cumcount() + 1)
        tx = self._inject_quality_issues(tx)
        log.info("transaction lines: %d", len(tx))

        ground_truth = dict(
            note="Simulation parameters. NOT used by any model - only to validate what models recover.",
            category_elasticity={c: s["elasticity"] for c, s in CATEGORY_SPECS.items()},
            product_elasticity=dict(zip(products["product_id"], products["true_elasticity"].round(3))),
            complements=[dict(a=a, b=b, prob=p) for a, b, p in COMPLEMENTS],
            incidents=INCIDENTS,
            personas=[p.__dict__ for p in PERSONAS],
        )
        personas = customers[["customer_id", "_persona", "_churn_date"]].rename(
            columns={"_persona": "persona", "_churn_date": "true_churn_date"})
        customers = customers.drop(columns=[c for c in customers if c.startswith("_")])
        products = products.drop(columns=["popularity", "true_elasticity"])
        products["current_list_price"] = prices[-1]
        promos_out = promos.assign(status=np.where(promos["start_date"] > self.dates[-1], "planned", "completed"))
        return dict(transactions=tx, customers=customers, products=products,
                    promotions=promos_out, ground_truth_personas=personas,
                    _ground_truth=ground_truth)

    def _inject_quality_issues(self, tx: pd.DataFrame) -> pd.DataFrame:
        """Realistic export problems the cleaning step must handle."""
        rng = self.rng
        n = len(tx)
        tx = tx.copy()
        miss = rng.choice(n, size=int(n * 0.004), replace=False)
        tx.loc[miss, "channel"] = np.nan
        neg = rng.choice(n, size=18, replace=False)
        tx.loc[neg, "quantity"] = -tx.loc[neg, "quantity"]
        cents = rng.choice(n, size=12, replace=False)
        tx.loc[cents, "unit_price"] = (tx.loc[cents, "unit_price"] * 100).round(2)
        dups = tx.sample(frac=0.003, random_state=int(rng.integers(1e9)))
        tx = pd.concat([tx, dups]).sort_values(["order_date", "order_id", "line_number"])
        return tx.reset_index(drop=True)


def generate_and_save(cfg: Config) -> dict[str, pd.DataFrame]:
    cfg.ensure_dirs()
    data = SalesDataGenerator(cfg).generate()
    for name, df in data.items():
        if name == "_ground_truth":
            save_json(df, cfg.raw_dir / "ground_truth.json")
        else:
            df.to_csv(cfg.raw_dir / f"{name}.csv", index=False)
    sample_dir = cfg.data_dir / "sample"   # small, committed slice for browsing the schema
    sample_dir.mkdir(parents=True, exist_ok=True)
    data["transactions"].head(2000).to_csv(sample_dir / "transactions_sample.csv", index=False)
    for name in ("products", "promotions"):
        data[name].to_csv(sample_dir / f"{name}.csv", index=False)
    log.info("raw data written to %s", cfg.raw_dir)
    return data
