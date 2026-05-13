"""Test that the num_categories fix works end-to-end."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC = PROJECT_ROOT / 'data' / 'processed'
df_raw = pd.read_csv(PROC / 'cleaned_apple_sales_enriched_realistic.csv')

# ---- Section 3 preprocessing (the fixed version) ----
df_raw['sale_date'] = pd.to_datetime(df_raw['sale_date'])
df_raw['year']  = df_raw['sale_date'].dt.year
df_raw['month'] = df_raw['sale_date'].dt.month

if 'country_norm_mapped' not in df_raw.columns:
    df_raw['country_norm_mapped'] = df_raw['country'].str.lower().str.strip()

monthly_aggs = {
    'sales_amount_realistic': 'sum',
    'quantity_realistic': 'sum',
    'price_realistic': 'mean',
    'store_name': 'first',
    'country_norm_mapped': 'first',
    'num_transactions': 'sum' if 'num_transactions' in df_raw.columns else 'count',
}
if 'num_unique_products' in df_raw.columns:
    monthly_aggs['num_unique_products'] = 'max'
if 'promo_flag' in df_raw.columns:
    monthly_aggs['promo_flag'] = 'mean'
for ecol in ['gdp_per_capita','inflation_rate','exchange_rate','internet_usage_pct']:
    if ecol in df_raw.columns:
        monthly_aggs[ecol] = 'first'
if 'num_categories' in df_raw.columns:
    monthly_aggs['num_categories'] = 'max'
# THE FIX: compute num_categories from category_id
if 'category_id' in df_raw.columns and 'num_categories' not in df_raw.columns:
    monthly_aggs['category_id'] = 'nunique'

if 'num_transactions' not in df_raw.columns:
    monthly_aggs.pop('num_transactions', None)

df_monthly = (
    df_raw.groupby(['store_id','year','month'])
    .agg(monthly_aggs)
    .reset_index()
)

df_monthly['date'] = pd.to_datetime(df_monthly[['year','month']].assign(day=1))
df_monthly.sort_values(['store_id','date'], inplace=True)
df_monthly.reset_index(drop=True, inplace=True)

if 'num_transactions' not in df_monthly.columns:
    df_monthly['num_transactions'] = (
        df_raw.groupby(['store_id','year','month']).size().values
    )
if 'num_unique_products' not in df_monthly.columns:
    df_monthly['num_unique_products'] = (
        df_raw.groupby(['store_id','year','month'])['product_id']
        .nunique().values
    )
# THE FIX: rename category_id -> num_categories
if 'category_id' in df_monthly.columns and 'num_categories' not in df_monthly.columns:
    df_monthly.rename(columns={'category_id': 'num_categories'}, inplace=True)

print(f"Monthly dataset: {df_monthly.shape[0]:,} rows x {df_monthly.shape[1]} columns")
print(f"num_categories in df_monthly.columns: {'num_categories' in df_monthly.columns}")
if 'num_categories' in df_monthly.columns:
    print(f"num_categories sample values: {df_monthly['num_categories'].head(5).tolist()}")
else:
    print("ERROR: num_categories NOT found in columns!")
    print(f"Available columns: {list(df_monthly.columns)}")
