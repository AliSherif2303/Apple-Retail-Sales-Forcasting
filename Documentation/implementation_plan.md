# Recommendation Report: Simulation Column Fixes

## Executive Summary

After auditing the full pipeline (especially [002_preprocess_externals.ipynb](file:///c:/Users/Ali%20Sherif/Apple-Retail-Sales-Forcasting/notebooks/002_preprocess_externals.ipynb) Cell 31), I identified **8 major issues** causing the data to look unrealistic. The root cause is that the simulation factors were designed for the *old* dataset and are now stale/misaligned after our product redistribution and date shift.

---

## Problem Diagnosis: Column-by-Column

### 1. `season_factor` -- TOO SIMPLE

**Source:** Cell 31, lines 190-200

```python
def seasonal_multiplier(month):
    if month == 11: return 1.3
    elif month == 12: return 1.5
    elif month == 1: return 0.8
    else: return 1.0  # <-- 8 months are IDENTICAL
```

**Problem:** 8 out of 12 months have exactly `1.0`. This creates a nearly flat seasonality with only Nov/Dec/Jan showing any variation. Real Apple sales show:
- **Sep spike** (new iPhone launch month)
- **Jun dip** (pre-announcement lull)
- **Mar bump** (spring event, iPad/Mac launches)
- **Jul-Aug dip** (summer lull before fall launches)

**Current distribution:**

| Value | Count | Months |
|---|---|---|
| 0.8 | 93,629 | Jan only |
| 1.0 | 824,574 | Feb-Oct (8 months!) |
| 1.3 | 78,754 | Nov only |
| 1.5 | 71,961 | Dec only |

> [!CAUTION]
> This is the **#1 reason** the seasonality heatmap looks flat. Feb-Oct are indistinguishable.

**Recommended fix:**
```python
SEASON_MAP = {
    1:  0.85,   # Post-holiday hangover
    2:  0.80,   # Lowest month (short + post-holiday)
    3:  0.95,   # Spring event (iPad/Mac launches)
    4:  0.90,   # Quiet
    5:  0.88,   # Pre-WWDC lull
    6:  0.92,   # WWDC hype
    7:  0.87,   # Summer lull
    8:  0.85,   # Pre-iPhone lull
    9:  1.25,   # iPhone launch month!
    10: 1.15,   # iPhone continued demand
    11: 1.30,   # Black Friday
    12: 1.45,   # Holiday season peak
}
```

---

### 2. `economic_factor` -- FROZEN IN TIME

**Source:** Cell 31, lines 206-211

```python
df['economic_factor'] = (
    (df['gdp_per_capita'] / df['gdp_per_capita'].mean()) *
    (1 - df['inflation_rate'] / 100)
).clip(0.5, 1.5)
```

**Problem:** The GDP and inflation data are **per-country-per-year** but they were computed using the old dataset's mean. After our date shift (+1 year), the `economic_factor` values are still the **same stale numbers** from the original 2020-2024 data. They don't reflect 2025 economic conditions at all.

Additionally, the formula divides by the *global* GDP mean, which mixes wealthy countries (US: $64K) with developing ones (Mexico: $8.8K), creating extreme spread.

**Recommended fix:**
- Recompute using the **same formula** but with updated GDP/inflation data for 2021-2025
- Normalize per-country instead of globally:
```python
# Per-country normalization (much more realistic)
df['economic_factor'] = df.groupby('country_norm_mapped').apply(
    lambda g: (g['gdp_per_capita'] / g['gdp_per_capita'].mean()) *
              (1 - g['inflation_rate'] / 100)
).values.clip(0.5, 1.5)
```

---

### 3. `promo_flag` / `promo_factor` -- PURELY RANDOM

**Source:** Cell 31, lines 217-218

```python
df['promo_flag'] = np.random.choice([0,1], size=len(df), p=[0.9,0.1])
df['promo_factor'] = np.where(df['promo_flag']==1, 1.4, 1.0)
```

**Problem:** Promotions are assigned with a flat 10% probability across ALL rows, regardless of:
- **Month** (Black Friday = Nov, Back-to-School = Aug/Sep should have more promos)
- **Product lifecycle** (older products get discounted more)
- **Category** (Accessories get more promos than iPhones)

**Current:** 106,926 promo rows (10.0%) uniformly spread.

**Recommended fix:**
```python
# Month-aware promo probabilities
PROMO_RATE_BY_MONTH = {
    1: 0.05, 2: 0.05, 3: 0.08, 4: 0.06,
    5: 0.07, 6: 0.10, 7: 0.12, 8: 0.15,  # Back-to-school
    9: 0.08, 10: 0.10, 11: 0.25, 12: 0.20  # Holiday promos
}

# Also: older products get more promos
# product_age > 365 days -> +10% promo chance
# product_age > 730 days -> +20% promo chance
```

---

### 4. `price_realistic` -- OVERSIMPLIFIED

**Source:** Cell 31, lines 225-234

```python
df['price_realistic'] = df['price'] * (1 + df['inflation_rate']/100)
df.loc[df['promo_flag']==1, 'price_realistic'] *= 0.85
df['price_realistic'] *= np.random.normal(1, 0.03, len(df))
```

**Then in our redistribution (Cell 6 of redistribute_products.py), we OVERWROTE this with:**
```python
noise = np.random.normal(0.97, 0.05, len(df)).clip(0.85, 1.10)
df['price_realistic'] = df['price'] * noise
```

> [!WARNING]
> We lost the inflation adjustment and promo discount logic during redistribution! The current `price_realistic` is just `price * random_noise` with no economic meaning.

**Recommended fix:** Reapply the full formula:
```python
# 1. Inflation adjustment
df['price_realistic'] = df['price'] * (1 + df['inflation_rate'] / 100)

# 2. Product age depreciation (older products sell cheaper)
age_discount = np.where(df['product_age_days'] > 365, 0.90,
               np.where(df['product_age_days'] > 730, 0.80, 1.0))
df['price_realistic'] *= age_discount

# 3. Promo discount
df.loc[df['promo_flag']==1, 'price_realistic'] *= 0.85

# 4. Small market noise
df['price_realistic'] *= np.random.normal(1, 0.03, len(df))
```

---

### 5. `product_trend` / `trend_factor` -- STALE

**Source:** Cell 31, lines 242-250

```python
product_trend_map = {
    pid: np.random.uniform(-0.0005, 0.0008)
    for pid in df['product_id'].unique()
}
df['product_trend'] = df['product_id'].map(product_trend_map)
df['trend_factor'] = 1 + df['product_trend'] * df['days_from_start']
```

**Problem:** The `product_trend` values were generated for the **old product IDs** (P-1 through P-89). After redistribution, we now have a completely different set of product IDs. The `product_trend` column is a **stale copy** that doesn't correspond to the new products.

Also, `days_from_start` was recalculated correctly from 2021-01-01, but the trend_factor uses the old product_trend values.

**Recommended fix:** Regenerate product_trend for the new product IDs:
```python
# New product trends based on category (some categories trend up, others down)
CATEGORY_TREND = {
    'CAT-4': 0.0003,   # Smartphones: slight uptrend
    'CAT-1': 0.0002,   # Laptops: steady
    'CAT-3': -0.0001,  # Tablets: slight decline
    'CAT-5': 0.0004,   # Wearables: growing fast
    'CAT-2': 0.0001,   # Audio: steady
    'CAT-7': 0.0000,   # Desktop: flat
    'CAT-8': 0.0005,   # Subscriptions: strong growth
    'CAT-6': -0.0002,  # Streaming devices: declining
    'CAT-9': -0.0003,  # Smart speakers: declining
    'CAT-10': 0.0001,  # Accessories: steady
}

# Add per-product noise on top of category trend
product_trend_map = {}
for _, row in products.iterrows():
    base = CATEGORY_TREND.get(row['Category_ID'], 0)
    noise = np.random.uniform(-0.0002, 0.0002)
    product_trend_map[row['Product_ID']] = base + noise
```

---

### 6. `store_factor` -- OK BUT STATIC

**Source:** Cell 31, lines 256-262

```python
store_effect_map = {sid: np.random.normal(1, 0.15) for sid in df['store_id'].unique()}
df['store_factor'] = df['store_id'].map(store_effect_map).clip(0.6, 1.5)
```

**Status:** This is actually fine -- store performance variation is realistic. However, it's **completely static** (a store has the same factor forever). Real stores have yearly performance variations.

**Optional enhancement:** Add year-over-year store performance drift:
```python
# Small random annual drift (+/- 5%)
store_year_drift = np.random.normal(1, 0.05)
```

---

### 7. `mu_demand` -- CASCADING ERRORS

**Source:** Cell 31, lines 268-290

```python
df['mu_demand'] = (
    df['quantity'] * df['season_factor'] * df['economic_factor'] *
    df['promo_factor'] * df['trend_factor'] * df['store_factor']
)
# Plus price elasticity adjustment
```

**Problem:** `mu_demand` multiplies ALL the broken factors together, compounding every issue above. Since `season_factor` is flat for 8 months, `economic_factor` is stale, `promo_factor` is random, and `trend_factor` uses old product trends -- the resulting `mu_demand` produces unrealistic demand patterns.

**Recommended fix:** This column should be **completely recomputed** after fixing all upstream factors.

---

### 8. Shock Event -- WRONG DATES

**Source:** Cell 31, lines 296-301

```python
shock_mask = (
    (df['sale_date'] >= '2022-03-01') &
    (df['sale_date'] <= '2022-06-01')
)
df.loc[shock_mask, 'mu_demand'] *= 0.7
```

**Problem:** This was meant to simulate the 2022 supply chain crisis, but:
1. After our +1 year shift, the data's 2022 is actually the old 2021 data
2. The shock was applied to the old data, then the dates shifted, so the shock is now effectively in the wrong place
3. We need to decide: should we model a 2022 shock in the new timeline? Or a different event?

**Recommended fix:** Apply shock events that make sense for the new 2021-2025 timeline:
```python
# 2022 Q1-Q2: Global supply chain crisis + Ukraine war
shock_2022 = (df['sale_date'] >= '2022-03-01') & (df['sale_date'] <= '2022-06-30')
df.loc[shock_2022, 'mu_demand'] *= 0.75

# 2020 COVID impact (not in our range, but 2021 Q1 still has aftereffects)
shock_2021 = (df['sale_date'] >= '2021-01-01') & (df['sale_date'] <= '2021-03-31')
df.loc[shock_2021, 'mu_demand'] *= 0.85
```

---

## Priority Order for Fixes

| Priority | Column | Impact | Effort |
|---|---|---|---|
| **P0** | `season_factor` | Fixes the flat seasonality heatmap | Easy |
| **P0** | `price_realistic` | Restores inflation/promo/age pricing logic | Easy |
| **P1** | `promo_flag/factor` | Adds realistic seasonal promotions | Medium |
| **P1** | `product_trend/trend_factor` | Fixes stale product lifecycle trends | Medium |
| **P1** | `mu_demand` + `quantity_realistic` | Must be recomputed after all upstream fixes | Medium |
| **P2** | `economic_factor` | Recompute with per-country normalization | Medium |
| **P2** | Shock events | Align to correct 2021-2025 timeline | Easy |
| **P3** | `store_factor` | Add annual drift (optional polish) | Easy |

---

## Proposed Implementation

> [!IMPORTANT]
> All fixes should be applied in a single new script `scripts/fix_simulation_factors.py` that reads `cleaned_apple_sales_v2.csv`, applies all fixes in order, and saves to `cleaned_apple_sales_v3.csv`.

The fixes must be applied **in dependency order**:
1. Fix `season_factor` (independent)
2. Fix `economic_factor` (independent)
3. Fix `promo_flag` / `promo_factor` (depends on season)
4. Fix `price_realistic` (depends on promo, inflation)
5. Fix `product_trend` / `trend_factor` (independent)
6. Recompute `mu_demand` (depends on ALL above)
7. Apply shock events (modifies mu_demand)
8. Apply price elasticity (modifies mu_demand)
9. Regenerate `quantity_realistic` via Negative Binomial
10. Recompute `sales_amount_realistic`

### Verification
- Re-run `analysis_report.py` on v3 to compare before/after
- The seasonality heatmap should show clear monthly variation
- Revenue should show Sep/Nov/Dec spikes
- Promo concentration in Q4 should be visible

## Open Questions

> [!IMPORTANT]
> **Q1:** Should the 2022 supply chain shock be kept, or would you prefer different/no shock events?

> [!IMPORTANT]
> **Q2:** For the `economic_factor`, should I fetch actual 2025 GDP/inflation estimates, or extrapolate from existing data?

> [!IMPORTANT]
> **Q3:** The current `quantity` column (original, not realistic) ranges 1-10 uniformly. It's used as the base for `mu_demand`. Should we keep it, or replace it with a category-aware base demand (e.g., iPhones sell more units than Mac Pros)?
