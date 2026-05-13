import pandas as pd
import numpy as np

v2 = pd.read_csv("data/processed/cleaned_apple_sales_v2.csv")
v3 = pd.read_csv("data/processed/cleaned_apple_sales_v3.csv")

print("=== REVENUE DECOMPOSITION ===")
r2 = v2["sales_amount_realistic"].sum()
r3 = v3["sales_amount_realistic"].sum()
print(f"v2 total rev: ${r2:,.0f}")
print(f"v3 total rev: ${r3:,.0f}")
print(f"Diff: ${r3-r2:,.0f} ({(r3-r2)/r2*100:+.1f}%)")

print("\n--- PRICE ---")
print(f"v2 price mean: ${v2['price'].mean():,.2f}  |  v3: ${v3['price'].mean():,.2f}")
print(f"v2 price_real mean: ${v2['price_realistic'].mean():,.2f}  |  v3: ${v3['price_realistic'].mean():,.2f}")
print(f"v2 price_real median: ${v2['price_realistic'].median():,.2f}  |  v3: ${v3['price_realistic'].median():,.2f}")
pr2 = v2['price_realistic'] / v2['price']
pr3 = v3['price_realistic'] / v3['price']
print(f"v2 price_real/price ratio: {pr2.mean():.4f}  |  v3: {pr3.mean():.4f}")

print("\n--- QUANTITY ---")
print(f"v2 qty_real mean: {v2['quantity_realistic'].mean():.3f}  |  v3: {v3['quantity_realistic'].mean():.3f}")
print(f"v2 qty_real median: {v2['quantity_realistic'].median():.1f}  |  v3: {v3['quantity_realistic'].median():.1f}")
z2 = (v2['quantity_realistic']==0).sum()
z3 = (v3['quantity_realistic']==0).sum()
print(f"v2 zeros: {z2:,} ({z2/len(v2)*100:.1f}%)  |  v3: {z3:,} ({z3/len(v3)*100:.1f}%)")

print("\n--- MU_DEMAND ---")
print(f"v2 mean: {v2['mu_demand'].mean():.3f}  |  v3: {v3['mu_demand'].mean():.3f}")
print(f"v2 median: {v2['mu_demand'].median():.3f}  |  v3: {v3['mu_demand'].median():.3f}")

print("\n--- ALL FACTORS ---")
for col in ['season_factor','economic_factor','promo_factor','trend_factor','store_factor']:
    m2 = v2[col].mean()
    m3 = v3[col].mean()
    s2 = v2[col].std()
    s3 = v3[col].std()
    print(f"{col:20s}: v2 mean={m2:.4f} std={s2:.4f}  |  v3 mean={m3:.4f} std={s3:.4f}")

print("\n--- QUANTITY BY CATEGORY ---")
cats = sorted(v2['category_name'].unique())
for cat in cats:
    q2 = v2[v2['category_name']==cat]['quantity_realistic'].mean()
    q3 = v3[v3['category_name']==cat]['quantity_realistic'].mean()
    r2c = v2[v2['category_name']==cat]['sales_amount_realistic'].sum()
    r3c = v3[v3['category_name']==cat]['sales_amount_realistic'].sum()
    chg = (r3c-r2c)/r2c*100 if r2c > 0 else 0
    print(f"{cat:25s}: qty {q2:.1f}->{q3:.1f} | rev ${r2c/1e6:.0f}M->${r3c/1e6:.0f}M ({chg:+.0f}%)")

print("\n--- YEARLY BREAKDOWN ---")
for y in range(2021, 2026):
    r2y = v2[v2['year']==y]['sales_amount_realistic'].sum()
    r3y = v3[v3['year']==y]['sales_amount_realistic'].sum()
    q2y = v2[v2['year']==y]['quantity_realistic'].mean()
    q3y = v3[v3['year']==y]['quantity_realistic'].mean()
    p2y = v2[v2['year']==y]['price_realistic'].mean()
    p3y = v3[v3['year']==y]['price_realistic'].mean()
    print(f"{y}: rev ${r2y/1e9:.2f}B->${r3y/1e9:.2f}B | qty {q2y:.1f}->{q3y:.1f} | price ${p2y:.0f}->${p3y:.0f}")

print("\n--- SHOCK IMPACT ---")
# 2022 Q1-Q2
s2 = v2[(v2['year']==2022)&(v2['month'].between(2,6))]['sales_amount_realistic'].mean()
s3 = v3[(v3['year']==2022)&(v3['month'].between(2,6))]['sales_amount_realistic'].mean()
n2 = v2[(v2['year']==2022)&(v2['month'].between(7,12))]['sales_amount_realistic'].mean()
n3 = v3[(v3['year']==2022)&(v3['month'].between(7,12))]['sales_amount_realistic'].mean()
print(f"2022 shock period (Feb-Jun):   v2=${s2:,.0f}  v3=${s3:,.0f}  ({(s3-s2)/s2*100:+.0f}%)")
print(f"2022 normal period (Jul-Dec):  v2=${n2:,.0f}  v3=${n3:,.0f}  ({(n3-n2)/n2*100:+.0f}%)")

# Revenue = qty * price, so decompose
print("\n--- REVENUE DECOMPOSITION (multiplicative) ---")
qty_effect = v3['quantity_realistic'].mean() / v2['quantity_realistic'].mean()
price_effect = v3['price_realistic'].mean() / v2['price_realistic'].mean()
print(f"Qty effect: {qty_effect:.4f} ({(qty_effect-1)*100:+.1f}%)")
print(f"Price effect: {price_effect:.4f} ({(price_effect-1)*100:+.1f}%)")
print(f"Combined: {qty_effect*price_effect:.4f} ({(qty_effect*price_effect-1)*100:+.1f}%)")
print(f"Actual rev ratio: {v3['sales_amount_realistic'].sum()/v2['sales_amount_realistic'].sum():.4f}")
