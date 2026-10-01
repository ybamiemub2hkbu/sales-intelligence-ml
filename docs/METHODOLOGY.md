# Methodology

How each module works, why it was built that way, and how it is validated. All tunable values live in `salesml/config.py`.

## 0. Data simulation (`salesml/data/generator.py`)

The simulator produces a line-level transaction export for a retailer with 8 categories and 89 products over 2023–2025.

* **Customers** get one of five hidden personas (loyalist, regular, deal seeker, occasional, one-and-done). Each persona has an order rate, extra-items rate, promotion sensitivity and mean lifetime. Individual rates vary with a Gamma(4, 0.25) multiplier. Lifetimes are exponential, and a customer's purchase intensity fades linearly to 30% over the 120 days before they churn. That fade is the signal a churn model can learn.
* **Order timing:** for each customer, the number of orders in their active window is Poisson with a mean equal to their rate times the summed daily intensity. Days are sampled in proportion to intensity, which multiplies weekly, yearly and holiday effects, promotion traffic (scaled by the persona's sensitivity) and a 10-day post-event dip after site-wide sales.
* **Category choice** multiplies the customer's Dirichlet category affinity (tilted by age band), category seasonality, regional taste and a promotion boost `(1 + 2 x discount) ^ sensitivity`.
* **Product choice** within a category is proportional to `popularity x (price / base price) ^ elasticity`, with product elasticities drawn around the category value. List prices change every 45–150 days by ±14% with mild inflation.
* **Complements**: 16 directed product pairs add the second item with a fixed probability.
* **Returns** depend on category, discount depth (≥25%) and channel (online higher).
* **Incidents and dirty data** are injected (see README). Ground truth is written to `data/raw/ground_truth.json` and is only used for validation.

## 1. Cleaning (`salesml/data/cleaning.py`)

Schema check, then: exact duplicates dropped; duplicate keys de-duplicated; non-positive quantities sign-corrected; unit prices about 100x the expected value divided by 100 (cents error); other price outliers reset to `list x (1 - discount)`; missing channels imputed with the customer's modal channel; unknown products dropped. Revenue, cost, discount amount, net revenue (zero if returned) and gross profit are then **recomputed** from clean inputs. Each fix is counted in `reports/tables/data_quality_report.csv`.

## 2. Point-in-time features (`salesml/features.py`)

`customer_snapshot(tx, as_of)` uses only transactions with `order_date < as_of`. It builds recency, frequency, monetary value, AOV, tenure, 90-day and prior-90-day order counts, order trend, 90/365-day spend, average inter-purchase gap, recency relative to that gap, discount share and depth, category breadth, digital and mobile share, return rate, items per order, promotion order share, region, age band and loyalty flag. A unit test changes all future rows and asserts the features stay identical.

## 3. Demand forecasting (`salesml/models/forecasting.py`)

* **Target:** weekly net revenue per category (complete weeks only).
* **Incident repair:** a week is replaced by its centred 9-week median when it deviates by more than 0.30 in log terms *and* by more than 4 robust standard deviations of that category's own noise, *and* no promotion or holiday explains it. This stops the February 2025 stock-out from being learned as seasonality. Evaluation always uses the unrepaired actuals.
* **Model:** one `HistGradientBoostingRegressor` across all categories, with category as a native categorical feature. Trees cannot extrapolate trend, so the target is `log(y_t) - log(level_t)`, where the level is the trailing 13-week mean. Features: recent lag ratios, last year's same-week (and neighbouring-week) ratio to its own level, momentum, YoY level change, planned category discount, site-wide promotion days (this year and last year's same week), and calendar and holiday flags. Forecasts are recursive.
* **Validation:** 4 rolling origins, 13 weeks apart (covering the final year), each trained only on prior data. Baselines: seasonal naive × trailing YoY growth, and an 8-week moving average. Metric: WAPE at category-week and total-week level, plus bias.
* **Selection:** the production method is whichever of {ML, ML + seasonal-naive ensemble} wins the backtest. The ensemble won here.
* **Intervals:** the 10th and 90th percentiles of backtest log errors, pooled by horizon bucket (1–4, 5–8, 9–13 weeks), applied to the point forecast.

## 4. Segmentation (`salesml/models/segmentation.py`)

Customers with no order in 365 days are assigned **Lapsed / Hibernating** by rule, because they would otherwise dominate the cluster geometry. Active customers are clustered with K-Means on standardised (log where skewed) recency, frequency, monetary value, AOV, discount share, category breadth and tenure. Channel share is excluded on purpose: it is bimodal and would split customers by channel rather than value. k is chosen in 4–6 by silhouette score on a 5,000-customer sample. Clusters are named by rules on their medians (highest spend → Champions; highest discount share → Deal Hunters; lowest tenure → New & Promising; and so on). Each segment gets a playbook action. Quintile RFM scores are also exported.

## 5. Churn (`salesml/models/churn.py`)

* **Population:** customers active in the last 365 days at the snapshot. **Label:** no order in the next 120 days.
* **Training:** three stacked snapshots (240, 300 and 360 days before the data end). **Test:** a snapshot 120 days before the end, so its outcome window is entirely after every training label.
* **Models:** logistic regression (signed-log + standardise), random forest and histogram gradient boosting, compared with 4-fold GroupKFold by customer (one customer can appear in several snapshots). Selection is by CV ROC-AUC; we report out-of-time ROC-AUC, PR-AUC, Brier score, calibration and decile lift.
* **Drivers:** permutation importance (drop in test ROC-AUC).
* **Targeting:** rank by `p(churn) x margin at stake`, where margin at stake = trailing-365-day revenue × horizon/365 × gross margin. Expected net profit of contacting the top k = Σ churned × save rate × margin − cost × k. Its argmax on the test snapshot sets the contact depth; that share of today's active customers forms the retention list.

## 6. CLV (`salesml/models/clv.py`)

Target: net revenue in the next 180 days, for customers active at the snapshot. Same out-of-time design as churn. Candidates: the finance run-rate rule (trailing 12-month revenue × 180/365), a Tweedie GLM (power 1.5, log link) and Poisson-loss gradient boosting. Metrics: MAE, RMSE, Spearman rank correlation, top-decile capture and total bias. Scores are crossed with churn probability into a **value × risk** matrix (top-quartile CLV × churn ≥ 30%).

## 7. Price elasticity and pricing (`salesml/models/elasticity.py`)

Per category, on the weekly product panel:

`log(units_pw) = e · log(effective price_pw) + product FE + week FE + ε`

Week fixed effects absorb traffic, seasonality and category-wide promotions (the same discount applies to every product in a category). Identification therefore comes from **cross-product list-price changes within a week**. The two-way demeaned estimator is solved by alternating projections. 95% CIs come from 200 week-clustered bootstrap resamples, with resampled weeks relabelled so duplicates keep separate fixed effects.

Pricing: with constant elasticity e, profit `(p − c) · q0 · (p/p0)^e` is maximised at `p* = c · e / (1 + e)` for e < −1. Recommendations are clipped to ±10% of today's list price (the range the model has observed), rounded to .99 endings, and valued with the last 12 weeks' volume.

## 8. Market basket (`salesml/models/basket.py`)

Exact pairwise rules from an order self-join: support, confidence and lift, with a minimum of 25 co-occurrences. Strong rules: lift ≥ 2 and confidence ≥ 10%. Rules are computed at product level (bundles) and category level (cross-shopping heatmap), and each bundle gets the average order value of orders that contain it.

## 9. Anomaly detection (`salesml/models/anomaly.py`)

43 series are monitored: total, 3 channels, 8 categories, 4 regions, 12 region × channel combinations and the top 15 products. For each:

1. Expected log revenue from gradient boosting on calendar, holiday, promotion and trend features, with **out-of-fold predictions holding out whole months** (fold = month index mod 5, so the same calendar month falls in different folds in different years). Shuffled K-fold would let a multi-day incident leak into training.
2. Rolling 3-day actual vs expected, as a log ratio with a level-based pseudo-count, turned into a robust z-score (median/MAD).
3. A second pass refits without days flagged in the first pass, so one incident cannot distort another year's baseline.
4. Days with |z| > 3.5 outside known holidays and the pre-Christmas rush are flagged. Flags within 2 days of each other are merged into incidents, and incidents moving less than 15% of an average day's total revenue are dropped as immaterial.

An Isolation Forest over the vector of all series' z-scores flags days with an unusual cross-series pattern.

## 10. Promotion ROI (`salesml/models/promotions.py`)

A category-day model (gradient boosting on calendar, holiday, pre-Christmas, Black Friday distance, trend and category) is trained only on **clean** category-days: not promoted in that category and not inside a site-wide event or its 10-day aftermath. Predictions on promoted days are rescaled so clean days are unbiased on average. For each event:

* incremental revenue = actual − counterfactual
* incremental GP = actual GP − counterfactual revenue × the category's normal margin
* post-event effect = the same comparison over the 10 days after
* ROI = (incremental GP + post-event effect) / discount given

Verdicts: ROI ≥ 0.5 **Scale**, 0–0.5 **Optimise**, < 0 **Redesign / stop**. Next year's planned calendar is annotated with each event's historical verdict.

## 11. Reporting and dashboard

`salesml/reporting.py` reads every module's JSON results, builds quantified recommendations ranked by gross-profit impact and writes `reports/EXECUTIVE_SUMMARY.md`. `salesml/dashboard.py` embeds the same results into one HTML file. The retention and price simulators recompute in the browser.
