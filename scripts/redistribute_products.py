"""
redistribute_products.py
========================
Replaces product data in the sales CSV with real Apple products,
shifts dates from 2020-2024 to 2021-2025, and extends data through Dec 31, 2025.

Distribution Strategy (Recency-Weighted Decay):
  60% current-year products | 20% Y-1 | 10% Y-2 | 5% older | 5% subscriptions
"""
import pandas as pd
import numpy as np
import string
import os

np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────
BASE = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting"
SALES   = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_enriched_realistic.csv")
PRODS   = os.path.join(BASE, "data", "processed", "generated_products.csv")
CATS    = os.path.join(BASE, "data", "raw", "category.csv")
OUTPUT  = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v2.csv")

START   = pd.Timestamp("2021-01-01")
W = {"cur": 0.60, "p1": 0.20, "p2": 0.10, "old": 0.05, "sub": 0.05}

print("=" * 60)
print("PRODUCT REDISTRIBUTION & DATE EXTENSION")
print("=" * 60)

# ── 1. Load ──────────────────────────────────────────────────
print("\n[1/7] Loading data...")
df   = pd.read_csv(SALES)
prod = pd.read_csv(PRODS)
cats = pd.read_csv(CATS)
print(f"  Sales: {len(df):,} rows | Products: {len(prod)} | Cats: {len(cats)}")

prod["Launch_Date"] = pd.to_datetime(prod["Launch_Date"])
prod["Launch_Year"] = prod["Launch_Date"].dt.year
cat_map = dict(zip(cats["category_id"], cats["category_name"]))

hw   = prod[prod["Category_ID"] != "CAT-8"].reset_index(drop=True)
subs = prod[(prod["Category_ID"] == "CAT-8") &
            (prod["Product_Name"] != "AppleCare One")].reset_index(drop=True)
subs_with_ac = prod[prod["Category_ID"] == "CAT-8"].reset_index(drop=True)
print(f"  Hardware: {len(hw)} | Subs (excl AppleCare): {len(subs)} | Subs (all): {len(subs_with_ac)}")

# ── 2. Shift dates +1 year ──────────────────────────────────
print("\n[2/7] Shifting dates +1 year...")
df["sale_date"] = pd.to_datetime(df["sale_date"]) + pd.DateOffset(years=1)
df["year"]  = df["sale_date"].dt.year
df["month"] = df["sale_date"].dt.month
print(f"  Range: {df['sale_date'].min().date()} -> {df['sale_date'].max().date()}")
for y, c in df.groupby("year").size().items():
    print(f"    {y}: {c:,}")

# ── 3. Generate new rows (Nov 13 – Dec 31, 2025) ────────────
print("\n[3/7] Generating new rows...")
AVG_NOV, AVG_DEC = 17_700, 18_000
existing_nov = len(df[(df["year"] == 2025) & (df["month"] == 11)])
need_nov = max(0, AVG_NOV - existing_nov)
need_dec = AVG_DEC
total_new = need_nov + need_dec
print(f"  Need: {need_nov:,} Nov + {need_dec:,} Dec = {total_new:,} new rows")

src = df[df["year"] == 2025]
ctx_cols = ["sale_id","store_id","store_name","city","country_norm_mapped",
            "quantity","exchange_rate","inflation_rate","internet_usage_pct",
            "gdp_type","gdp_per_capita","season_factor","economic_factor",
            "promo_flag","promo_factor","store_factor","mu_demand",
            "quantity_realistic","product_trend","trend_factor"]

idx = np.random.choice(src.index, total_new, replace=True)
new = src.loc[idx, ctx_cols].reset_index(drop=True)

# Unique sale IDs
pref = ["".join(np.random.choice(list(string.ascii_uppercase), 2)) for _ in range(total_new)]
nums = np.random.randint(1000, 999999, total_new)
new["sale_id"] = [f"{p}-{n}" for p, n in zip(pref, nums)]

# Dates
nov_d = pd.date_range("2025-11-13", "2025-11-30")
dec_d = pd.date_range("2025-12-01", "2025-12-31")
dates = np.concatenate([np.random.choice(nov_d, need_nov),
                        np.random.choice(dec_d, need_dec)])
new["sale_date"] = dates
new["year"]  = pd.to_datetime(new["sale_date"]).dt.year
new["month"] = pd.to_datetime(new["sale_date"]).dt.month

for c in ["product_id","product_name","category_id","category_name",
          "launch_date","price","sales_amount","invalid_launch_flag",
          "product_age_days","days_from_start","price_realistic",
          "sales_amount_realistic"]:
    new[c] = np.nan

df = pd.concat([df, new], ignore_index=True)
print(f"  Total rows now: {len(df):,}")

# ── 4. Build product pools ──────────────────────────────────
print("\n[4/7] Building product pools...")

def build_pool(year):
    """Returns (product_df_with_weight) for weighted sampling."""
    tiers = [
        (hw[hw["Launch_Year"] == year],     W["cur"]),
        (hw[hw["Launch_Year"] == year - 1], W["p1"]),
        (hw[hw["Launch_Year"] == year - 2], W["p2"]),
        (hw[hw["Launch_Year"] <= year - 3], W["old"]),
    ]
    # Only keep non-empty tiers
    active = [(t, w) for t, w in tiers if len(t) > 0]
    s = subs_with_ac if year >= 2025 else subs
    total_w = sum(w for _, w in active) + W["sub"]

    parts = []
    for t, w in active:
        t2 = t.copy()
        t2["_w"] = (w / total_w) / len(t2)
        parts.append(t2)
    s2 = s.copy()
    s2["_w"] = (W["sub"] / total_w) / len(s2) if len(s2) > 0 else 0
    parts.append(s2)

    pool = pd.concat(parts, ignore_index=True)
    pool["_w"] = pool["_w"] / pool["_w"].sum()  # normalise
    return pool

pools = {}
for y in range(2021, 2026):
    pools[y] = build_pool(y)
    print(f"  {y}: {len(pools[y])} products in pool")

# ── 5. Assign products ──────────────────────────────────────
print("\n[5/7] Assigning products (vectorised)...")
for year in range(2021, 2026):
    mask = df["year"] == year
    n = mask.sum()
    p = pools[year]
    sel = p.iloc[np.random.choice(len(p), n, p=p["_w"].values)]
    df.loc[mask, "product_id"]    = sel["Product_ID"].values
    df.loc[mask, "product_name"]  = sel["Product_Name"].values
    df.loc[mask, "category_id"]   = sel["Category_ID"].values
    df.loc[mask, "launch_date"]   = sel["Launch_Date"].dt.strftime("%Y-%m-%d").values
    df.loc[mask, "price"]         = sel["Price"].astype(float).values
    df.loc[mask, "category_name"] = sel["Category_ID"].map(cat_map).values
    print(f"  {year}: {n:,} rows assigned")

# ── 6. Recalculate derived columns ──────────────────────────
print("\n[6/7] Recalculating derived columns...")
df["sale_date"]   = pd.to_datetime(df["sale_date"])
df["launch_date"] = pd.to_datetime(df["launch_date"])
df["price"]       = df["price"].astype(float)

df["sales_amount"]       = df["quantity"] * df["price"]
df["product_age_days"]   = (df["sale_date"] - df["launch_date"]).dt.days
df["invalid_launch_flag"]= df["launch_date"] > df["sale_date"]
df["days_from_start"]    = (df["sale_date"] - START).dt.days

noise = np.random.normal(0.97, 0.05, len(df)).clip(0.85, 1.10)
df["price_realistic"]         = df["price"] * noise
df["sales_amount_realistic"]  = df["quantity_realistic"] * df["price_realistic"]

df["sale_date"]   = df["sale_date"].dt.strftime("%Y-%m-%d")
df["launch_date"] = df["launch_date"].dt.strftime("%Y-%m-%d")
print("  [OK] All derived columns recalculated")

# ── 7. Save ──────────────────────────────────────────────────
print("\n[7/7] Saving...")
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
print(f"  [OK] Saved to {OUTPUT}")

# ── Verification ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
print(f"Rows:     {len(df):,}")
print(f"Range:    {df['sale_date'].min()} -> {df['sale_date'].max()}")
print(f"Products: {df['product_name'].nunique()}")
print(f"\nNaN check:")
for c in ["product_id","product_name","price","sales_amount_realistic"]:
    print(f"  {c}: {df[c].isna().sum()}")
print(f"\nYear dist:")
for y, c in df.groupby("year").size().items():
    print(f"  {y}: {c:,}")
print(f"\nCategory dist:")
for cat, c in df.groupby("category_name").size().sort_values(ascending=False).items():
    print(f"  {cat}: {c:,} ({c/len(df)*100:.1f}%)")
print("\n[DONE] Script completed successfully!")
