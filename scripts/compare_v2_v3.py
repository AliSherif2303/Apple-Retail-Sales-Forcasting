"""
compare_v2_v3.py
================
Side-by-side comparison of v2 (before fixes) vs v3 (after fixes).
Generates comparison charts saved to data/processed/comparison_plots/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings
warnings.filterwarnings("ignore")

BASE = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting"
V2   = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v2.csv")
V3   = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v3.csv")
OUT  = os.path.join(BASE, "data", "processed", "comparison_plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9", "xtick.color": "#8b949e",
    "ytick.color": "#8b949e", "grid.color": "#21262d",
    "font.family": "sans-serif", "font.size": 10,
})
C_V2 = "#f85149"  # Red for v2
C_V3 = "#3fb950"  # Green for v3
MNAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

print("Loading datasets...")
v2 = pd.read_csv(V2); v2["sale_date"] = pd.to_datetime(v2["sale_date"])
v3 = pd.read_csv(V3); v3["sale_date"] = pd.to_datetime(v3["sale_date"])
print(f"  v2: {len(v2):,} rows | v3: {len(v3):,} rows")

# ================================================================
# 1. SEASONALITY COMPARISON (Monthly avg revenue)
# ================================================================
print("[1/8] Seasonality comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
for ax, df, label, color in [(axes[0], v2, "v2 (Before)", C_V2),
                              (axes[1], v3, "v3 (After)", C_V3)]:
    monthly = df.groupby("month")["sales_amount_realistic"].mean()
    ax.bar(range(1, 13), monthly.values, color=color, alpha=0.8)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MNAMES, rotation=45, fontsize=8)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_ylabel("Avg Sales ($)")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(monthly.values):
        ax.text(i+1, v+50, f"${v:,.0f}", ha="center", fontsize=7)
fig.suptitle("Seasonality: Monthly Average Revenue", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_seasonality_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 2. SEASON FACTOR DISTRIBUTION
# ================================================================
print("[2/8] Season factor distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
v2_sf = v2.groupby("month")["season_factor"].first()
v3_sf = v3.groupby("month")["season_factor"].first()
axes[0].bar(range(1,13), v2_sf.values, color=C_V2, alpha=0.8)
axes[0].set_title("v2: season_factor (4 values)", fontsize=12, fontweight="bold")
axes[0].set_xticks(range(1,13)); axes[0].set_xticklabels(MNAMES, rotation=45, fontsize=8)
axes[0].set_ylim(0.5, 1.7)
axes[0].axhline(1.0, color="#8b949e", linestyle="--", alpha=0.5)
for i,v in enumerate(v2_sf.values): axes[0].text(i+1, v+0.02, f"{v:.2f}", ha="center", fontsize=8)

axes[1].bar(range(1,13), v3_sf.values, color=C_V3, alpha=0.8)
axes[1].set_title("v3: season_factor (12 values)", fontsize=12, fontweight="bold")
axes[1].set_xticks(range(1,13)); axes[1].set_xticklabels(MNAMES, rotation=45, fontsize=8)
axes[1].set_ylim(0.5, 1.7)
axes[1].axhline(1.0, color="#8b949e", linestyle="--", alpha=0.5)
for i,v in enumerate(v3_sf.values): axes[1].text(i+1, v+0.02, f"{v:.2f}", ha="center", fontsize=8)
fig.suptitle("Season Factor: Before vs After", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_season_factor.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 3. YEARLY REVENUE TREND
# ================================================================
print("[3/8] Yearly revenue trend...")
fig, ax = plt.subplots(figsize=(12, 5))
yr_v2 = v2.groupby("year")["sales_amount_realistic"].sum()
yr_v3 = v3.groupby("year")["sales_amount_realistic"].sum()
x = np.arange(len(yr_v2))
w = 0.35
ax.bar(x - w/2, yr_v2.values, w, color=C_V2, label="v2 (Before)", alpha=0.85)
ax.bar(x + w/2, yr_v3.values, w, color=C_V3, label="v3 (After)", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(yr_v2.index)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e9:.2f}B"))
ax.set_title("Total Revenue by Year", fontsize=14, fontweight="bold")
ax.legend(framealpha=0.3, edgecolor="#30363d")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_yearly_revenue.png"), dpi=150)
plt.close()

# ================================================================
# 4. PROMO DISTRIBUTION BY MONTH
# ================================================================
print("[4/8] Promo distribution...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, df, label, color in [(axes[0], v2, "v2: Flat 10%", C_V2),
                              (axes[1], v3, "v3: Seasonal Promos", C_V3)]:
    promo_rate = df.groupby("month")["promo_flag"].mean() * 100
    ax.bar(range(1, 13), promo_rate.values, color=color, alpha=0.8)
    ax.set_xticks(range(1,13)); ax.set_xticklabels(MNAMES, rotation=45, fontsize=8)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Promo Rate (%)")
    ax.set_ylim(0, 35)
    for i,v in enumerate(promo_rate.values): ax.text(i+1, v+0.5, f"{v:.0f}%", ha="center", fontsize=8)
fig.suptitle("Promotion Rate by Month", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_promo_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 5. CATEGORY DEMAND (avg qty)
# ================================================================
print("[5/8] Category demand comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, df, label, color in [(axes[0], v2, "v2: Uniform Demand", C_V2),
                              (axes[1], v3, "v3: Category-Aware", C_V3)]:
    cat_qty = df.groupby("category_name")["quantity_realistic"].mean().sort_values(ascending=True)
    ax.barh(range(len(cat_qty)), cat_qty.values, color=color, alpha=0.8)
    ax.set_yticks(range(len(cat_qty))); ax.set_yticklabels(cat_qty.index, fontsize=9)
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Avg Quantity per Transaction")
    for i,v in enumerate(cat_qty.values): ax.text(v+0.1, i, f"{v:.1f}", va="center", fontsize=8)
fig.suptitle("Average Quantity by Category", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_category_demand.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 6. DAILY REVENUE TIME SERIES
# ================================================================
print("[6/8] Daily revenue time series...")
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
for ax, df, label, color in [(axes[0], v2, "v2 (Before)", C_V2),
                              (axes[1], v3, "v3 (After)", C_V3)]:
    daily = df.groupby("sale_date")["sales_amount_realistic"].sum()
    daily.index = pd.to_datetime(daily.index)
    roll = daily.rolling(30).mean()
    ax.plot(daily.index, daily.values, color=color, alpha=0.2, linewidth=0.5)
    ax.plot(roll.index, roll.values, color=color, linewidth=2, label="30-day MA")
    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.1f}M"))
    ax.legend(framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
fig.suptitle("Daily Revenue: Before vs After", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_daily_revenue.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 7. SEASONALITY HEATMAP COMPARISON
# ================================================================
print("[7/8] Seasonality heatmaps...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, df, label in [(axes[0], v2, "v2 (Before)"), (axes[1], v3, "v3 (After)")]:
    pivot = df.groupby(["year", "month"])["sales_amount_realistic"].mean().unstack()
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index)
    ax.set_xticks(range(12)); ax.set_xticklabels(MNAMES, fontsize=7, rotation=45)
    ax.set_title(label, fontsize=12, fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                c = "white" if val > pivot.values[~np.isnan(pivot.values)].max() * 0.6 else "#c9d1d9"
                ax.text(j, i, f"${val/1000:.0f}K", ha="center", va="center", fontsize=6, color=c)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Seasonality Heatmap: Avg Revenue by Month & Year", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_heatmap_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 8. SHOCK EVENTS VISIBILITY
# ================================================================
print("[8/8] Shock event visibility...")
fig, ax = plt.subplots(figsize=(16, 5))
for df, label, color in [(v2, "v2", C_V2), (v3, "v3", C_V3)]:
    monthly = df.groupby(df["sale_date"].dt.to_period("M"))["sales_amount_realistic"].sum()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, color=color, linewidth=2, label=label, alpha=0.85)

# Highlight shock periods
import matplotlib.patches as mpatches
ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-31"), alpha=0.1, color="#d29922", label="COVID aftermath")
ax.axvspan(pd.Timestamp("2022-02-01"), pd.Timestamp("2022-06-30"), alpha=0.1, color="#f85149", label="Supply crisis")
ax.axvspan(pd.Timestamp("2023-01-01"), pd.Timestamp("2023-03-31"), alpha=0.1, color="#bc8cff", label="Tech pullback")

ax.set_title("Monthly Revenue with Shock Events Highlighted", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1e6:.0f}M"))
ax.legend(framealpha=0.3, edgecolor="#30363d", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_shock_events.png"), dpi=150)
plt.close()

# ================================================================
# STATISTICAL COMPARISON TABLE
# ================================================================
print("\n" + "=" * 60)
print("STATISTICAL COMPARISON: v2 vs v3")
print("=" * 60)

def stats(df, name):
    return {
        "Dataset": name,
        "Total Revenue": f"${df['sales_amount_realistic'].sum():,.0f}",
        "Mean Transaction": f"${df['sales_amount_realistic'].mean():,.2f}",
        "Median Transaction": f"${df['sales_amount_realistic'].median():,.2f}",
        "Mean Qty": f"{df['quantity_realistic'].mean():.2f}",
        "Zero Qty %": f"{(df['quantity_realistic']==0).mean()*100:.1f}%",
        "Mean Price Real.": f"${df['price_realistic'].mean():,.2f}",
        "Promo Rate": f"{df['promo_flag'].mean()*100:.1f}%",
        "Season Factor Unique": str(df['season_factor'].nunique()),
        "Econ Factor Std": f"{df['economic_factor'].std():.3f}",
        "Trend Factor Std": f"{df['trend_factor'].std():.3f}",
    }

s2, s3 = stats(v2, "v2"), stats(v3, "v3")
print(f"\n{'Metric':<25s} {'v2 (Before)':<22s} {'v3 (After)':<22s}")
print("-" * 70)
for key in s2:
    if key == "Dataset": continue
    print(f"{key:<25s} {s2[key]:<22s} {s3[key]:<22s}")

# Monthly comparison
print(f"\n{'Month':<8s} {'v2 Avg Rev':<15s} {'v3 Avg Rev':<15s} {'Change':<10s}")
print("-" * 50)
for m in range(1, 13):
    a2 = v2[v2["month"]==m]["sales_amount_realistic"].mean()
    a3 = v3[v3["month"]==m]["sales_amount_realistic"].mean()
    chg = ((a3 - a2) / a2) * 100
    print(f"{MNAMES[m-1]:<8s} ${a2:>10,.0f}    ${a3:>10,.0f}    {chg:>+.0f}%")

# Category comparison
print(f"\n{'Category':<25s} {'v2 Avg Qty':<12s} {'v3 Avg Qty':<12s}")
print("-" * 50)
cats = sorted(v2["category_name"].unique())
for cat in cats:
    q2 = v2[v2["category_name"]==cat]["quantity_realistic"].mean()
    q3 = v3[v3["category_name"]==cat]["quantity_realistic"].mean()
    print(f"{cat:<25s} {q2:>8.1f}    {q3:>8.1f}")

print(f"\nPlots saved to: {OUT}")
print("[DONE] Comparison complete!")
