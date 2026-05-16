# 5.2 Data Preprocessing

## Table of Contents
1. [Data Loading & Merging](#1-data-loading--merging)
2. [Data Cleaning](#2-data-cleaning)
3. [Feature Engineering — Simulation Pipeline](#3-feature-engineering--simulation-pipeline)
4. [Product Redistribution & Date Extension](#4-product-redistribution--date-extension)
5. [Simulation Factor Rebuild (v2 → v3)](#5-simulation-factor-rebuild-v2--v3)
6. [Monthly Aggregation & ML Feature Engineering](#6-monthly-aggregation--ml-feature-engineering)
7. [Train-Test Split & Cross-Validation](#7-train-test-split--cross-validation)
8. [Dataset Version Summary](#8-dataset-version-summary)

---

## 1. Data Loading & Merging

**Notebook:** `notebooks/01_data_loading_and_merging.ipynb`

### 1.1 Raw Data Sources

| File | Records | Description |
|------|---------|-------------|
| `sales.csv` | ~1,040,200 | Transactions (sale_id, store_id, product_id, quantity, sale_date) |
| `products.csv` | 89 | Product catalog with prices and launch dates |
| `stores.csv` | 75 | Apple Stores — name, city, country |
| `category.csv` | 10 | Product categories (CAT-1 to CAT-10) |
| `warranty.csv` | 30,000 | Warranty claims and repair statuses |

### 1.2 External Macroeconomic Data

| File | Description |
|------|-------------|
| `gdp.csv` | GDP per capita by country/year (World Bank) |
| `infilation.csv` | Annual inflation rates |
| `exchange.csv` | Currency exchange rates |
| `Individuals using the Internet.csv` | Internet penetration % |
| `Year-Category-Product-StartingPrice.csv` | Historical Apple product pricing |

### 1.3 Merge Pipeline

```python
# Step 1: Load core tables
df = pd.read_csv("sales.csv")
products = pd.read_csv("products.csv")
stores = pd.read_csv("stores.csv")
categories = pd.read_csv("category.csv")

# Step 2: Join sales → products → stores → categories
df = df.merge(products, on="product_id", how="left")
df = df.merge(stores, on="store_id", how="left")
df = df.merge(categories, on="category_id", how="left")

# Step 3: Enrich with macroeconomic indicators (country × year join)
df = df.merge(gdp, on=["country_norm_mapped", "year"], how="left")
df = df.merge(inflation, on=["country_norm_mapped", "year"], how="left")
df = df.merge(exchange, on=["country_norm_mapped", "year"], how="left")
df = df.merge(internet, on=["country_norm_mapped", "year"], how="left")

# Output: cleaned_apple_sales_enriched.csv (208 MB)
```

**Result:** `cleaned_apple_sales_enriched.csv` — ~1,040,200 rows × 24 columns.

---

## 2. Data Cleaning

**Notebook:** `notebooks/002_preprocess_externals.ipynb`

### 2.1 GDP Anomaly Detection & Correction

South Korea and Taiwan had GDP values >200,000 — these were **Total GDP** instead of **GDP per capita**.

```python
# Flag suspicious values
df['gdp_type'] = np.where(df['gdp'] > 200_000, 'GDP_total', 'GDP_per_capita')

# Manual correction with real per-capita values
manual_gdp_pc = pd.DataFrame({
    "country_norm_mapped": ["korea, rep."]*5 + ["taiwan"]*5,
    "year": [2020,2021,2022,2023,2024]*2,
    "gdp_per_capita": [
        31728, 34998, 32394, 33147, 34000,   # Korea
        28371, 33775, 32756, 33907, 35000     # Taiwan
    ]
})
df = df.merge(manual_gdp_pc, on=["country_norm_mapped","year"], how="left")
df.loc[df['gdp'] > 200_000, 'gdp_cleaned'] = df['gdp_per_capita']
```

### 2.2 Missing Value Treatment

```python
# internet_usage_pct: fill with grouped mean (country × year)
df['internet_usage_pct'] = df.groupby(
    ['country_norm_mapped', 'year']
)['internet_usage_pct'].transform(lambda x: x.fillna(x.mean()))

# Fallback: fill remaining NaNs with country-level mean
df['internet_usage_pct'] = df.groupby(
    'country_norm_mapped'
)['internet_usage_pct'].transform(lambda x: x.fillna(x.mean()))
```

### 2.3 Column Cleanup

```python
df.drop(columns=['gdp', 'gdp_cleaned', 'country', 'country_norm'], inplace=True)
df['gdp_per_capita'] = df['gdp_cleaned']  # rename to standard name
```

### 2.4 Duplicate & Outlier Handling

- No duplicate `sale_id` entries were found in the raw data.
- Outlier handling is applied later during demand simulation via the 99.5th percentile cap on `quantity_realistic`.
- Zero-inflation (2%) is intentionally injected to simulate real-world zero-sale days.

---

## 3. Feature Engineering — Simulation Pipeline

**Notebook:** `notebooks/002_preprocess_externals.ipynb` (Cell 31+)

This pipeline creates 12 derived columns that simulate realistic Apple retail demand patterns. The original version had simplified logic that was later improved in the v3 rebuild.

### 3.1 Season Factor (Original)

```python
def seasonal_multiplier(month):
    if month == 11: return 1.3    # Black Friday
    elif month == 12: return 1.5  # Holiday season
    elif month == 1: return 0.8   # Post-holiday dip
    else: return 1.0              # Baseline

df['season_factor'] = df['sale_date'].dt.month.apply(seasonal_multiplier)
# Result: Only 4 distinct values (0.8, 1.0, 1.3, 1.5)
```

### 3.2 Economic Factor (Original)

```python
df['economic_factor'] = (
    (df['gdp_per_capita'] / df['gdp_per_capita'].mean()) *  # Global mean normalization
    (1 - df['inflation_rate'] / 100)
)
df['economic_factor'] = df['economic_factor'].clip(0.5, 1.5)
```

### 3.3 Promotion System (Original)

```python
df['promo_flag'] = np.random.choice([0,1], size=len(df), p=[0.9, 0.1])  # Flat 10%
df['promo_factor'] = np.where(df['promo_flag']==1, 1.4, 1.0)  # Fixed 40% boost
```

### 3.4 Dynamic Pricing (Original)

```python
df['price_realistic'] = df['price'] * (1 + df['inflation_rate']/100)
df.loc[df['promo_flag']==1, 'price_realistic'] *= 0.85   # 15% promo discount
df['price_realistic'] *= np.random.normal(1, 0.03, len(df))  # Market noise
```

### 3.5 Product Lifecycle Trend (Original)

```python
df['days_from_start'] = (df['sale_date'] - df['sale_date'].min()).dt.days
product_trend_map = {pid: np.random.uniform(-0.0005, 0.0008)
                     for pid in df['product_id'].unique()}
df['product_trend'] = df['product_id'].map(product_trend_map)
df['trend_factor'] = (1 + df['product_trend'] * df['days_from_start']).clip(0.5, 1.8)
```

### 3.6 Store Heterogeneity (Original)

```python
store_effect_map = {sid: np.random.normal(1, 0.15)
                    for sid in df['store_id'].unique()}
df['store_factor'] = df['store_id'].map(store_effect_map).clip(0.6, 1.5)
```

### 3.7 Base Mean Demand (mu_demand)

```python
df['mu_demand'] = (
    df['quantity'] * df['season_factor'] * df['economic_factor'] *
    df['promo_factor'] * df['trend_factor'] * df['store_factor']
).clip(lower=0.1)
```

### 3.8 Price Elasticity

```python
elasticity = -0.8
price_change_ratio = (df['price_realistic'] - df['price']) / df['price']
df['mu_demand'] *= (1 + elasticity * price_change_ratio)
df['mu_demand'] = df['mu_demand'].clip(lower=0.1)
```

### 3.9 Shock Event (2022 Supply Chain Crisis)

```python
shock_mask = (df['sale_date'] >= '2022-03-01') & (df['sale_date'] <= '2022-06-01')
df.loc[shock_mask, 'mu_demand'] *= 0.7  # 30% demand drop
```

### 3.10 Negative Binomial Demand Generation

```python
k = 2.8  # Dispersion parameter (higher = less variance)
p = k / (k + df['mu_demand'])
df['quantity_realistic'] = np.random.negative_binomial(k, p)
```

### 3.11 Zero Inflation & Outlier Capping

```python
zero_mask = np.random.choice([0,1], size=len(df), p=[0.98, 0.02])
df.loc[zero_mask==1, 'quantity_realistic'] = 0

upper_cap = df['quantity_realistic'].quantile(0.995)
df['quantity_realistic'] = df['quantity_realistic'].clip(upper=upper_cap)
```

### 3.12 Final Revenue

```python
df['sales_amount_realistic'] = df['quantity_realistic'] * df['price_realistic']
```

**Output:** `cleaned_apple_sales_enriched_realistic.csv` (356 MB)

---

## 4. Product Redistribution & Date Extension

**Script:** `scripts/redistribute_products.py`

This script replaces placeholder product IDs with 98 real Apple products and shifts the date range.

### 4.1 Date Shift (+1 Year)

```python
df["sale_date"] = pd.to_datetime(df["sale_date"]) + pd.DateOffset(years=1)
# 2020-2024 → 2021-2025
```

### 4.2 New Row Generation (Nov–Dec 2025)

```python
# Generate ~35,700 new rows for Nov 13 – Dec 31, 2025
need_nov = max(0, 17_700 - existing_nov)  # ~17,700 target
need_dec = 18_000
# Sample context from existing 2025 rows, assign new sale_ids and dates
```

### 4.3 Recency-Weighted Product Distribution

Products are assigned using a decay-weighted sampling strategy:

| Weight | Pool | Description |
|--------|------|-------------|
| 60% | Current year | Products launched in the sale year |
| 20% | Y−1 | Previous year products |
| 10% | Y−2 | Two years old |
| 5% | Older | Three+ years old |
| 5% | Subscriptions | Apple Music, TV+, Arcade, iCloud+, Fitness+ |

```python
def build_pool(year):
    tiers = [
        (hw[hw["Launch_Year"] == year],     0.60),  # Current
        (hw[hw["Launch_Year"] == year - 1], 0.20),   # Y-1
        (hw[hw["Launch_Year"] == year - 2], 0.10),   # Y-2
        (hw[hw["Launch_Year"] <= year - 3], 0.05),   # Older
    ]
    # Add subscriptions at 5%
    # Normalize weights and return weighted pool
```

### 4.4 Derived Column Recalculation

```python
df["sales_amount"]        = df["quantity"] * df["price"]
df["product_age_days"]    = (df["sale_date"] - df["launch_date"]).dt.days
df["invalid_launch_flag"] = df["launch_date"] > df["sale_date"]
df["days_from_start"]     = (df["sale_date"] - pd.Timestamp("2021-01-01")).dt.days
```

**Output:** `cleaned_apple_sales_v2.csv` (366 MB, ~1,068,918 rows)

---

## 5. Simulation Factor Rebuild (v2 → v3)

**Script:** `scripts/v3 & v2 datasets fixing.py`

After product redistribution, the original simulation factors (season, economic, promo, etc.) became stale because they were computed on the old product assignments. This script rebuilds all 12 simulation columns in 4 phases.

### Phase 1: Independent Factors

#### 5.1 Season Factor (Improved: 12 distinct monthly values)

```python
SEASON_MAP = {
    1: 0.85,  2: 0.80,  3: 0.95,  4: 0.90,
    5: 0.88,  6: 0.92,  7: 0.87,  8: 0.90,
    9: 1.25, 10: 1.15, 11: 1.30, 12: 1.45
}
df["season_factor"] = df["month"].map(SEASON_MAP)
# Before: 4 distinct values → After: 12 distinct values
```

#### 5.2 Economic Factor (Per-country normalization)

```python
country_means = df.groupby("country_norm_mapped")["gdp_per_capita"].transform("mean")
gdp_ratio = df["gdp_per_capita"] / country_means
inflation_effect = 1 - df["inflation_rate"] / 100
df["economic_factor"] = (gdp_ratio * inflation_effect).clip(0.6, 1.4)
```

#### 5.3 Product Trend (Category-aware lifecycle)

```python
CATEGORY_TREND = {
    "CAT-4":  0.0003,  # Smartphones: growing
    "CAT-5":  0.0004,  # Wearables: growing fast
    "CAT-8":  0.0005,  # Subscriptions: fastest growth
    "CAT-6": -0.0002,  # Streaming devices: declining
    "CAT-9": -0.0003,  # Smart Speakers: declining
    # ... other categories
}
# Each product gets category base + random noise ±0.0002
```

#### 5.4 Store Factor (Annual drift)

```python
for store in df["store_id"].unique():
    for year in range(2021, 2026):
        drift = np.random.normal(1, 0.03)  # ±3% annual drift
        df.loc[mask, "store_factor"] = np.clip(base * drift, 0.6, 1.5)
```

### Phase 2: Dependent Factors

#### 5.5 Promotion System (3-layer model)

```python
# Layer 1: Monthly base rate (seasonal promotions)
MONTHLY_PROMO_RATE = {
    1: 0.05, 2: 0.05, 3: 0.08, ..., 11: 0.25, 12: 0.20
}

# Layer 2: Product age boost (older products get more promos)
age_boost = np.where(df["product_age_days"] > 730, 0.15,   # >2yr: +15%
            np.where(df["product_age_days"] > 365, 0.08, 0.00))  # >1yr: +8%

# Layer 3: Category boost (accessories, audio get extra promos)
cat_boost_map = {"CAT-10": 0.10, "CAT-2": 0.05, "CAT-9": 0.08}

promo_prob = (base_promo + age_boost + cat_boost).clip(0, 0.40)
df["promo_flag"] = (np.random.random(len(df)) < promo_prob).astype(int)
df["promo_factor"] = np.where(df["promo_flag"]==1,
                              np.random.uniform(1.15, 1.50, len(df)), 1.0)
# Result: ~15.8% overall promo rate (vs flat 10% before)
```

#### 5.6 Price Realistic (4-layer pricing)

```python
# Layer 1: Inflation adjustment
df["price_realistic"] = df["price"] * (1 + df["inflation_rate"] / 100)

# Layer 2: Product lifecycle depreciation
age_factor = np.where(df["product_age_days"] > 1095, 0.80,   # >3yr: -20%
             np.where(df["product_age_days"] > 730,  0.88,    # >2yr: -12%
             np.where(df["product_age_days"] > 365,  0.95, 1.00)))  # >1yr: -5%
df["price_realistic"] *= age_factor

# Layer 3: Promo discount (15% off)
df.loc[df["promo_flag"]==1, "price_realistic"] *= 0.85

# Layer 4: Market noise (±2%)
df["price_realistic"] *= np.random.normal(1.0, 0.02, len(df))
```

### Phase 3: Demand Generation

#### 5.7 mu_demand (Category-aware base demand)

```python
CATEGORY_BASE = {
    "CAT-4": 1.4,  # Smartphones (highest demand)
    "CAT-8": 1.5,  # Subscriptions
    "CAT-7": 0.5,  # Desktops (lowest — expensive, low volume)
}

# Price-inverse scaling (cheaper → higher volume)
price_scale = np.where(df["price"] < 100,  2.0,
              np.where(df["price"] < 500,   1.3,
              np.where(df["price"] < 1000,  1.0,
              np.where(df["price"] < 2000,  0.7, 0.4))))

df["mu_demand"] = (
    df["quantity"] * cat_base * price_scale *
    df["season_factor"] * df["economic_factor"] *
    df["promo_factor"] * df["trend_factor"] * df["store_factor"]
).clip(lower=0.1)
```

#### 5.8 Price Elasticity (Tiered)

```python
elasticity = np.where(df["price"] < 500,  -1.2,   # Budget: very elastic
             np.where(df["price"] < 1500, -0.8,    # Mid-range
                                          -0.5))   # Premium: inelastic
price_change = (df["price_realistic"] - df["price"]) / df["price"]
df["mu_demand"] *= (1 + elasticity * price_change)
```

#### 5.9 Historical Shock Events

| Period | Event | Multiplier | Rows Affected |
|--------|-------|-----------|---------------|
| 2021 Q1 | COVID aftermath | ×0.85 | ~52,000 |
| 2022 Feb–Jun | Supply chain + Ukraine crisis | ×0.75 | ~86,000 |
| 2023 Q1 | Tech layoffs | ×0.90 | ~52,000 |
| 2024 Q4 (CAT-4 only) | iPhone 16 launch boost | ×1.15 | ~19,000 |

#### 5.10 Quantity Generation (Negative Binomial)

```python
k = 2.5  # Dispersion (higher = tighter distribution)
p_nb = k / (k + df["mu_demand"])
df["quantity_realistic"] = np.random.negative_binomial(k, p_nb)

# Zero inflation: 2% of transactions have zero quantity
zero_mask = np.random.random(len(df)) < 0.02
df.loc[zero_mask, "quantity_realistic"] = 0

# Outlier cap: 99.5th percentile
df["quantity_realistic"] = df["quantity_realistic"].clip(
    upper=df["quantity_realistic"].quantile(0.995))
```

### Phase 4: Final Revenue

```python
df["sales_amount_realistic"] = df["quantity_realistic"] * df["price_realistic"]
```

**Output:** `cleaned_apple_sales_v3.csv` (379 MB, ~1,068,918 rows, 35 columns)

---

## 6. Monthly Aggregation & ML Feature Engineering

**Notebooks:** `CAT_LIVE.ipynb`, `CATBOOST_regulized.ipynb`

Before model training, daily transaction data is aggregated to monthly store-level summaries and enriched with time-series features.

### 6.1 Monthly Aggregation

```python
df_monthly = df_raw.groupby(["store_id", "year", "month"]).agg({
    "sales_amount_realistic": "sum",
    "quantity_realistic": "sum",
    "price_realistic": "mean",
    "store_name": "first",
    "country_norm_mapped": "first",
    "promo_flag": "mean",
    "gdp_per_capita": "first",
    "inflation_rate": "first",
    "exchange_rate": "first",
    "internet_usage_pct": "first",
}).reset_index()

# Additional aggregates
df_monthly["num_transactions"]    = grp.size().values
df_monthly["num_unique_products"] = grp["product_id"].nunique().values
df_monthly["num_categories"]      = grp["category_id"].nunique().values
```

### 6.2 Lag Features

```python
SAFE_LAGS = [1, 2, 3, 6, 12]
for lag in SAFE_LAGS:
    df_monthly[f"sales_lag_{lag}"] = (
        df_monthly.groupby("store_id")["sales_amount_realistic"].shift(lag)
    )
```

### 6.3 Rolling Statistics

```python
for w in [3, 6]:
    rolled = df_monthly.groupby("store_id")["sales_amount_realistic"].shift(1).rolling(w, min_periods=1)
    df_monthly[f"sales_roll_mean_{w}"] = rolled.mean().reset_index(0, drop=True)
```

### 6.4 Momentum Features

```python
df_monthly["sales_mom_pct"] = (
    df_monthly.groupby("store_id")["sales_amount_realistic"].pct_change()
)
df_monthly["sales_lag1_vs_roll6"] = (
    df_monthly["sales_lag_1"] / df_monthly["sales_roll_mean_6"].replace(0, np.nan)
)
```

### 6.5 Cyclical Time Encoding

```python
df_monthly["month_sin"] = np.sin(2 * np.pi * df_monthly["month"] / 12)
df_monthly["month_cos"] = np.cos(2 * np.pi * df_monthly["month"] / 12)
df_monthly["is_holiday_season"] = df_monthly["month"].isin([11, 12]).astype(int)
df_monthly["is_launch_season"]  = df_monthly["month"].isin([9, 10]).astype(int)
```

### 6.6 Economic Change Features

```python
for col in ["gdp_per_capita", "inflation_rate", "exchange_rate", "internet_usage_pct"]:
    new = col.replace("_per_capita","").replace("_rate","").replace("_pct","") + "_change"
    df_monthly[new] = df_monthly.groupby("store_id")[col].pct_change().fillna(0)
```

### 6.7 Store Encoding

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_monthly["store_encoded"] = le.fit_transform(df_monthly["store_id"])
```

### 6.8 NaN Cleanup

```python
# Drop first 12 months per store (insufficient lag history)
df_monthly["row_num"] = df_monthly.groupby("store_id").cumcount()
df_clean = df_monthly[df_monthly["row_num"] >= 12].copy()
df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)
```

### Final Feature Set (28 features)

| Category | Features |
|----------|----------|
| Lag | sales_lag_1, sales_lag_2, sales_lag_3, sales_lag_6, sales_lag_12 |
| Rolling | sales_roll_mean_3, sales_roll_mean_6 |
| Momentum | sales_mom_pct, sales_lag1_vs_roll6 |
| Price/Promo | price_realistic, promo_flag |
| Cyclical Time | month_sin, month_cos, is_holiday_season, is_launch_season |
| Economic | gdp_per_capita, inflation_rate, exchange_rate, internet_usage_pct |
| Economic Change | gdp_change, inflation_change, exchange_change, internet_usage_change |
| Store | store_encoded, num_transactions, num_unique_products, num_categories |
| Calendar | year |

---

## 7. Train-Test Split & Cross-Validation

**Notebooks:** `CATBOOST_regulized.ipynb`, `CAT_LIVE.ipynb`

### 7.1 Temporal Hold-Out Split

A temporal (chronological) split is used instead of random splitting to prevent data leakage in time-series forecasting:

```python
# Last 5 months of data used as test set
cutoff_date = df_clean["date"].max() - pd.DateOffset(months=5)
train_mask = df_clean["date"] <= cutoff_date
test_mask  = df_clean["date"] > cutoff_date

X_train = df_clean.loc[train_mask, FEATURES]
y_train = df_clean.loc[train_mask, TARGET]
X_test  = df_clean.loc[test_mask, FEATURES]
y_test  = df_clean.loc[test_mask, TARGET]
```

### 7.2 TimeSeriesSplit Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# Used for hyperparameter tuning via GridSearchCV
cat_search = GridSearchCV(
    CatBoostRegressor(...),
    param_grid={...},
    cv=tscv,            # Respects temporal ordering
    scoring='neg_mae',
)
cat_search.fit(X_train, y_train)
```

**Why TimeSeriesSplit?** Standard K-Fold would allow future data to leak into training folds. TimeSeriesSplit ensures each fold's training set only contains data *before* the validation set, respecting the temporal ordering of sales data.

---

## 8. Dataset Version Summary

| Version | File | Size | Rows | Key Changes |
|---------|------|------|------|-------------|
| Base | `cleaned_apple_sales.csv` | 140 MB | ~1,040,200 | Merged sales + products + stores + categories |
| Enriched | `cleaned_apple_sales_enriched.csv` | 208 MB | ~1,040,200 | + GDP, inflation, exchange rate, internet |
| Realistic | `cleaned_apple_sales_enriched_realistic.csv` | 356 MB | ~1,040,200 | + 12 simulation factors (original) |
| V2 | `cleaned_apple_sales_v2.csv` | 366 MB | ~1,068,918 | + Real products, date shift, Nov-Dec 2025 |
| **V3 (Final)** | **`cleaned_apple_sales_v3.csv`** | **379 MB** | **~1,068,918** | **Rebuilt all simulation factors (4-phase)** |

### Final Dataset Schema (V3 — 35 columns)

| Column | Type | Description |
|--------|------|-------------|
| sale_id | string | Unique transaction ID |
| sale_date | date | Transaction date (2021-01-01 to 2025-12-31) |
| store_id | string | Store identifier (ST-1 to ST-75) |
| product_id | string | Product identifier |
| quantity | int | Original quantity from Kaggle |
| product_name | string | Real Apple product name (98 products) |
| launch_date | date | Product launch date |
| price | float | Original product price (USD) |
| store_name | string | Apple Store name |
| city | string | Store city |
| category_id | string | Category (CAT-1 to CAT-10) |
| category_name | string | Category name |
| sales_amount | float | quantity × price |
| invalid_launch_flag | bool | True if sale_date < launch_date |
| product_age_days | int | Days since product launch |
| exchange_rate | float | Country exchange rate for the year |
| inflation_rate | float | Country inflation rate for the year |
| internet_usage_pct | float | Country internet penetration % |
| country_norm_mapped | string | Normalized country name (lowercase) |
| year | int | Sale year |
| gdp_type | string | GDP_total or GDP_per_capita |
| gdp_per_capita | float | Corrected GDP per capita |
| season_factor | float | Monthly seasonality multiplier (12 values) |
| economic_factor | float | Per-country economic multiplier |
| promo_flag | int | 1 if promotion active |
| promo_factor | float | Promotion demand boost (1.15–1.50) |
| price_realistic | float | Simulated price with inflation + depreciation |
| days_from_start | int | Days since 2021-01-01 |
| product_trend | float | Category-aware lifecycle trend |
| trend_factor | float | Computed trend multiplier |
| store_factor | float | Store performance multiplier |
| mu_demand | float | Combined expected demand signal |
| quantity_realistic | int | Neg. Binomial sampled quantity |
| sales_amount_realistic | float | quantity_realistic × price_realistic |
| month | int | Sale month (1–12) |
