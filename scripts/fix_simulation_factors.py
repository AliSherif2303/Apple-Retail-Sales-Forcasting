"""
fix_simulation_factors.py
=========================
Fixes all simulation columns in cleaned_apple_sales_v2.csv to produce
more realistic Apple retail sales data. Saves as cleaned_apple_sales_v3.csv.

Phases:
  1. Independent factors  (season, economic, product_trend, store)
  2. Dependent factors    (promo, price_realistic, trend_factor)
  3. Demand generation    (mu_demand, shocks, elasticity, quantity)
  4. Final revenue        (sales_amount_realistic)
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

BASE = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting"
INPUT  = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v2.csv")
OUTPUT = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v3.csv")

print("=" * 60)
print("FIX SIMULATION FACTORS")
print("=" * 60)

# ── Load ─────────────────────────────────────────────────────
print("\n[LOAD] Reading data...")
df = pd.read_csv(INPUT)
df["sale_date"]   = pd.to_datetime(df["sale_date"])
df["launch_date"] = pd.to_datetime(df["launch_date"])
df["price"] = df["price"].astype(float)
print(f"  Rows: {len(df):,}")

# =============================================================
# PHASE 1 : Independent Factors
# =============================================================
print("\n" + "=" * 60)
print("PHASE 1: Independent Factors")
print("=" * 60)

# ── 1.1 season_factor ────────────────────────────────────────
print("\n[1.1] Fixing season_factor...")
SEASON_MAP = {
    1: 0.85,  2: 0.80,  3: 0.95,  4: 0.90,
    5: 0.88,  6: 0.92,  7: 0.87,  8: 0.90,
    9: 1.25, 10: 1.15, 11: 1.30, 12: 1.45
}
df["season_factor"] = df["month"].map(SEASON_MAP)
print("  Before: 4 distinct values -> After: 12 distinct values")
print("  Monthly factors:", dict(df.groupby("month")["season_factor"].first()))

# ── 1.2 economic_factor ──────────────────────────────────────
print("\n[1.2] Fixing economic_factor (per-country normalization)...")
country_means = df.groupby("country_norm_mapped")["gdp_per_capita"].transform("mean")
gdp_ratio = df["gdp_per_capita"] / country_means
inflation_effect = 1 - df["inflation_rate"] / 100
df["economic_factor"] = (gdp_ratio * inflation_effect).clip(0.6, 1.4)
print(f"  Mean: {df['economic_factor'].mean():.3f}")
print(f"  Std:  {df['economic_factor'].std():.3f}")
print(f"  Range: [{df['economic_factor'].min():.2f}, {df['economic_factor'].max():.2f}]")

# ── 1.3 product_trend ────────────────────────────────────────
print("\n[1.3] Fixing product_trend (category-aware)...")
CATEGORY_TREND = {
    "CAT-4":  0.0003,  "CAT-5":  0.0004,  "CAT-8":  0.0005,
    "CAT-1":  0.0002,  "CAT-2":  0.0001,  "CAT-3": -0.0001,
    "CAT-7":  0.0000,  "CAT-10": 0.0001,  "CAT-6": -0.0002,
    "CAT-9": -0.0003
}
trend_map = {}
for pid in df["product_id"].unique():
    cat = df.loc[df["product_id"] == pid, "category_id"].iloc[0]
    base = CATEGORY_TREND.get(cat, 0)
    trend_map[pid] = base + np.random.uniform(-0.0002, 0.0002)

df["product_trend"] = df["product_id"].map(trend_map)
print(f"  Unique trends: {df['product_trend'].nunique()}")
print(f"  Range: [{df['product_trend'].min():.6f}, {df['product_trend'].max():.6f}]")

# ── 1.4 store_factor (annual drift) ─────────────────────────
print("\n[1.4] Adding store_factor annual drift...")
base_store = df.groupby("store_id")["store_factor"].first().to_dict()
for store in df["store_id"].unique():
    base = base_store[store]
    for year in range(2021, 2026):
        mask = (df["store_id"] == store) & (df["year"] == year)
        drift = np.random.normal(1, 0.03)
        df.loc[mask, "store_factor"] = np.clip(base * drift, 0.6, 1.5)
print(f"  Unique store-year combos: {df.groupby(['store_id','year']).ngroups}")

# =============================================================
# PHASE 2 : Dependent Factors
# =============================================================
print("\n" + "=" * 60)
print("PHASE 2: Dependent Factors")
print("=" * 60)

# ── 2.1 trend_factor ─────────────────────────────────────────
print("\n[2.1] Recomputing trend_factor...")
df["trend_factor"] = (1 + df["product_trend"] * df["days_from_start"]).clip(0.5, 1.8)
print(f"  Mean: {df['trend_factor'].mean():.3f}")
print(f"  Std:  {df['trend_factor'].std():.3f}")

# ── 2.2 promo_flag / promo_factor ────────────────────────────
print("\n[2.2] Rebuilding promo system (seasonal + lifecycle + category)...")
MONTHLY_PROMO_RATE = {
    1: 0.05, 2: 0.05, 3: 0.08, 4: 0.06, 5: 0.07, 6: 0.10,
    7: 0.12, 8: 0.15, 9: 0.08, 10: 0.10, 11: 0.25, 12: 0.20
}
base_promo = df["month"].map(MONTHLY_PROMO_RATE)

# Age boost
age_boost = np.where(df["product_age_days"] > 730, 0.15,
            np.where(df["product_age_days"] > 365, 0.08, 0.00))

# Category boost
cat_boost_map = {"CAT-10": 0.10, "CAT-2": 0.05, "CAT-9": 0.08}
cat_boost = df["category_id"].map(cat_boost_map).fillna(0)

promo_prob = (base_promo + age_boost + cat_boost).clip(0, 0.40)
df["promo_flag"] = (np.random.random(len(df)) < promo_prob).astype(int)

# Variable promo factor (15-50% demand boost)
promo_factor_vals = np.random.uniform(1.15, 1.50, len(df))
df["promo_factor"] = np.where(df["promo_flag"] == 1, promo_factor_vals, 1.0)

promo_pct = df["promo_flag"].mean() * 100
print(f"  Overall promo rate: {promo_pct:.1f}%")
print("  Monthly promo rates:")
for m in range(1, 13):
    r = df[df["month"] == m]["promo_flag"].mean() * 100
    print(f"    Month {m:2d}: {r:.1f}%")

# ── 2.3 price_realistic ──────────────────────────────────────
print("\n[2.3] Rebuilding price_realistic (inflation + age + promo + noise)...")

# Inflation adjustment
df["price_realistic"] = df["price"] * (1 + df["inflation_rate"] / 100)

# Product lifecycle depreciation
age_factor = np.where(df["product_age_days"] > 1095, 0.80,
             np.where(df["product_age_days"] > 730,  0.88,
             np.where(df["product_age_days"] > 365,  0.95, 1.00)))
df["price_realistic"] *= age_factor

# Promo discount (15% off)
df.loc[df["promo_flag"] == 1, "price_realistic"] *= 0.85

# Market noise
df["price_realistic"] *= np.random.normal(1.0, 0.02, len(df))

# Floor for subscriptions
df.loc[df["category_id"] == "CAT-8", "price_realistic"] = \
    df.loc[df["category_id"] == "CAT-8", "price_realistic"].clip(lower=0.99)

print(f"  Mean price_realistic: ${df['price_realistic'].mean():,.2f}")
print(f"  Median: ${df['price_realistic'].median():,.2f}")

# =============================================================
# PHASE 3 : Demand Generation
# =============================================================
print("\n" + "=" * 60)
print("PHASE 3: Demand Generation")
print("=" * 60)

# ── 3.1 mu_demand (category-aware) ───────────────────────────
print("\n[3.1] Building mu_demand (category + price-inverse scaling)...")
CATEGORY_BASE = {
    "CAT-4": 1.4, "CAT-2": 1.3, "CAT-10": 1.2, "CAT-5": 1.1,
    "CAT-3": 0.9, "CAT-1": 0.7, "CAT-8": 1.5, "CAT-7": 0.5,
    "CAT-6": 0.8, "CAT-9": 0.6
}
cat_base = df["category_id"].map(CATEGORY_BASE)

price_scale = np.where(df["price"] < 100,  2.0,
              np.where(df["price"] < 500,   1.3,
              np.where(df["price"] < 1000,  1.0,
              np.where(df["price"] < 2000,  0.7, 0.4))))

df["mu_demand"] = (
    df["quantity"] * cat_base * price_scale *
    df["season_factor"] * df["economic_factor"] *
    df["promo_factor"] * df["trend_factor"] * df["store_factor"]
).clip(lower=0.1)

print(f"  mu_demand mean: {df['mu_demand'].mean():.2f}")
print(f"  mu_demand median: {df['mu_demand'].median():.2f}")

# ── 3.2 Price elasticity ─────────────────────────────────────
print("\n[3.2] Applying price elasticity...")
elasticity = np.where(df["price"] < 500, -1.2,
             np.where(df["price"] < 1500, -0.8, -0.5))
price_change = (df["price_realistic"] - df["price"]) / df["price"]
df["mu_demand"] *= (1 + elasticity * price_change)
df["mu_demand"] = df["mu_demand"].clip(lower=0.1)
print(f"  mu_demand after elasticity: mean={df['mu_demand'].mean():.2f}")

# ── 3.3 Shock events ─────────────────────────────────────────
print("\n[3.3] Applying shock events...")

# 2021 Q1: COVID aftermath
m1 = (df["sale_date"] >= "2021-01-01") & (df["sale_date"] <= "2021-03-31")
df.loc[m1, "mu_demand"] *= 0.85
print(f"  2021 Q1 COVID aftermath: {m1.sum():,} rows * 0.85")

# 2022 Q1-Q2: Supply chain + Ukraine
m2 = (df["sale_date"] >= "2022-02-01") & (df["sale_date"] <= "2022-06-30")
df.loc[m2, "mu_demand"] *= 0.75
print(f"  2022 Q1-Q2 supply crisis: {m2.sum():,} rows * 0.75")

# 2023 Q1: Tech layoffs
m3 = (df["sale_date"] >= "2023-01-01") & (df["sale_date"] <= "2023-03-31")
df.loc[m3, "mu_demand"] *= 0.90
print(f"  2023 Q1 tech pullback: {m3.sum():,} rows * 0.90")

# 2024 Q4: Strong iPhone 16 launch
m4 = ((df["sale_date"] >= "2024-09-01") & (df["sale_date"] <= "2024-12-31") &
      (df["category_id"] == "CAT-4"))
df.loc[m4, "mu_demand"] *= 1.15
print(f"  2024 Q4 iPhone 16 boost: {m4.sum():,} rows * 1.15")

# ── 3.4 quantity_realistic ───────────────────────────────────
print("\n[3.4] Generating quantity_realistic (Negative Binomial)...")
k = 2.5
p_nb = k / (k + df["mu_demand"])
df["quantity_realistic"] = np.random.negative_binomial(k, p_nb)

# Zero inflation (2%)
zero_mask = np.random.random(len(df)) < 0.02
df.loc[zero_mask, "quantity_realistic"] = 0

# Cap outliers
upper_cap = df["quantity_realistic"].quantile(0.995)
df["quantity_realistic"] = df["quantity_realistic"].clip(upper=upper_cap)
print(f"  Mean qty: {df['quantity_realistic'].mean():.2f}")
print(f"  Zeros: {(df['quantity_realistic']==0).sum():,} ({(df['quantity_realistic']==0).mean()*100:.1f}%)")

# =============================================================
# PHASE 4 : Final Revenue
# =============================================================
print("\n" + "=" * 60)
print("PHASE 4: Final Revenue")
print("=" * 60)

df["sales_amount"] = df["quantity"] * df["price"]
df["sales_amount_realistic"] = df["quantity_realistic"] * df["price_realistic"]

# Recalculate product_age_days and invalid_launch_flag
df["product_age_days"] = (df["sale_date"] - df["launch_date"]).dt.days
df["invalid_launch_flag"] = df["launch_date"] > df["sale_date"]

# Format dates
df["sale_date"]   = df["sale_date"].dt.strftime("%Y-%m-%d")
df["launch_date"] = df["launch_date"].dt.strftime("%Y-%m-%d")

print(f"  Total revenue: ${df['sales_amount_realistic'].sum():,.0f}")
print(f"  Mean transaction: ${df['sales_amount_realistic'].mean():,.2f}")
print(f"  Median: ${df['sales_amount_realistic'].median():,.2f}")

# ── Save ─────────────────────────────────────────────────────
print("\n[SAVE] Writing output...")
col_order = [
    "sale_id","sale_date","store_id","product_id","quantity",
    "product_name","launch_date","price","store_name","city",
    "category_id","category_name","sales_amount","invalid_launch_flag",
    "product_age_days","exchange_rate","inflation_rate","internet_usage_pct",
    "country_norm_mapped","year","gdp_type","gdp_per_capita",
    "season_factor","economic_factor","promo_flag","promo_factor",
    "price_realistic","days_from_start","product_trend","trend_factor",
    "store_factor","mu_demand","quantity_realistic","sales_amount_realistic","month"
]
df = df[col_order].sort_values("sale_date").reset_index(drop=True)
df.to_csv(OUTPUT, index=False)
print(f"  Saved to: {OUTPUT}")

# ── Verification ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Rows:     {len(df):,}")
print(f"Range:    {df['sale_date'].min()} -> {df['sale_date'].max()}")
print(f"Products: {df['product_name'].nunique()}")
print(f"\nYear distribution:")
for y, c in df.groupby("year").size().items():
    rev = df[df["year"]==y]["sales_amount_realistic"].sum()
    print(f"  {y}: {c:,} rows | ${rev:,.0f}")
print(f"\nMonthly avg revenue:")
for m in range(1, 13):
    avg = df[df["month"]==m]["sales_amount_realistic"].mean()
    print(f"  Month {m:2d}: ${avg:,.0f}")
print(f"\nCategory distribution:")
for cat, grp in df.groupby("category_name"):
    avg_qty = grp["quantity_realistic"].mean()
    print(f"  {cat:25s}: avg_qty={avg_qty:.1f}")
print(f"\nNaN check:")
for c in ["product_id","price_realistic","sales_amount_realistic","mu_demand"]:
    print(f"  {c}: {df[c].isna().sum()}")
print("\n[DONE] Fix script completed!")
