"""
analysis_report.py
==================
Generates comprehensive analysis plots for the redistributed Apple sales data.
Saves all charts to data/processed/analysis_plots/
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import os
import warnings
warnings.filterwarnings("ignore")

BASE = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting"
CSV  = os.path.join(BASE, "data", "processed", "cleaned_apple_sales_v2.csv")
OUT  = os.path.join(BASE, "data", "processed", "analysis_plots")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "font.family": "sans-serif",
    "font.size": 10,
})
PALETTE = ["#58a6ff","#3fb950","#d29922","#f85149","#bc8cff",
           "#79c0ff","#56d364","#e3b341","#ff7b72","#d2a8ff"]

print("Loading data...")
df = pd.read_csv(CSV)
df["sale_date"] = pd.to_datetime(df["sale_date"])
df["launch_date"] = pd.to_datetime(df["launch_date"])
print(f"  Loaded {len(df):,} rows")

# ================================================================
# 1. Monthly Transaction Volume Over Time
# ================================================================
print("[1/10] Monthly transaction volume...")
fig, ax = plt.subplots(figsize=(14, 5))
monthly = df.groupby(df["sale_date"].dt.to_period("M")).size()
monthly.index = monthly.index.to_timestamp()
ax.plot(monthly.index, monthly.values, color=PALETTE[0], linewidth=1.5)
ax.fill_between(monthly.index, monthly.values, alpha=0.15, color=PALETTE[0])
ax.set_title("Monthly Transaction Volume (2021-2025)", fontsize=14, fontweight="bold")
ax.set_ylabel("Transactions")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "01_monthly_volume.png"), dpi=150)
plt.close()

# ================================================================
# 2. Year-over-Year Comparison by Month
# ================================================================
print("[2/10] YoY monthly comparison...")
fig, ax = plt.subplots(figsize=(14, 5))
for i, year in enumerate(range(2021, 2026)):
    yd = df[df["year"] == year].groupby("month").size()
    ax.plot(yd.index, yd.values, color=PALETTE[i], label=str(year),
            linewidth=2, marker="o", markersize=4)
ax.set_title("Year-over-Year Monthly Transactions", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Transactions")
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.legend(framealpha=0.3, edgecolor="#30363d")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "02_yoy_monthly.png"), dpi=150)
plt.close()

# ================================================================
# 3. Category Distribution Over Years (Stacked Bar)
# ================================================================
print("[3/10] Category distribution...")
fig, ax = plt.subplots(figsize=(14, 6))
cat_year = df.groupby(["year", "category_name"]).size().unstack(fill_value=0)
cat_year_pct = cat_year.div(cat_year.sum(axis=1), axis=0) * 100
cat_year_pct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, width=0.7)
ax.set_title("Category Distribution by Year (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Percentage")
ax.set_xlabel("")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", framealpha=0.3, edgecolor="#30363d", fontsize=8)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "03_category_dist.png"), dpi=150)
plt.close()

# ================================================================
# 4. Top 10 Products by Revenue
# ================================================================
print("[4/10] Top products by revenue...")
fig, ax = plt.subplots(figsize=(14, 6))
top_rev = df.groupby("product_name")["sales_amount_realistic"].sum().nlargest(15)
bars = ax.barh(range(len(top_rev)), top_rev.values, color=PALETTE[0], alpha=0.85)
ax.set_yticks(range(len(top_rev)))
ax.set_yticklabels(top_rev.index, fontsize=9)
ax.set_title("Top 15 Products by Total Realistic Revenue", fontsize=14, fontweight="bold")
ax.set_xlabel("Revenue ($)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "04_top_products_revenue.png"), dpi=150)
plt.close()

# ================================================================
# 5. Sales Amount Distribution (Histogram)
# ================================================================
print("[5/10] Sales amount distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["sales_amount_realistic"].clip(0, 30000), bins=80, color=PALETTE[1], alpha=0.8, edgecolor="none")
axes[0].set_title("Sales Amount (Realistic) Distribution", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Amount ($)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(df["sales_amount_realistic"].mean(), color=PALETTE[3], linestyle="--", label=f"Mean: ${df['sales_amount_realistic'].mean():,.0f}")
axes[0].axvline(df["sales_amount_realistic"].median(), color=PALETTE[2], linestyle="--", label=f"Median: ${df['sales_amount_realistic'].median():,.0f}")
axes[0].legend(framealpha=0.3, edgecolor="#30363d")

axes[1].hist(df["quantity_realistic"].clip(0, 30), bins=30, color=PALETTE[4], alpha=0.8, edgecolor="none")
axes[1].set_title("Quantity (Realistic) Distribution", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Quantity")
axes[1].set_ylabel("Frequency")
axes[1].axvline(df["quantity_realistic"].mean(), color=PALETTE[3], linestyle="--", label=f"Mean: {df['quantity_realistic'].mean():.1f}")
axes[1].legend(framealpha=0.3, edgecolor="#30363d")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "05_distributions.png"), dpi=150)
plt.close()

# ================================================================
# 6. Seasonality Heatmap (Month x Year)
# ================================================================
print("[6/10] Seasonality heatmap...")
fig, ax = plt.subplots(figsize=(14, 5))
pivot = df.groupby(["year", "month"])["sales_amount_realistic"].mean().unstack()
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xticks(range(12))
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
ax.set_title("Average Sales Amount by Month & Year (Seasonality Heatmap)", fontsize=14, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Avg Sales ($)")
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"${val:,.0f}", ha="center", va="center", fontsize=7,
                    color="white" if val > pivot.values.max() * 0.6 else "#c9d1d9")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "06_seasonality_heatmap.png"), dpi=150)
plt.close()

# ================================================================
# 7. Price Distribution by Category (Box Plot)
# ================================================================
print("[7/10] Price by category...")
fig, ax = plt.subplots(figsize=(14, 6))
cats_sorted = df.groupby("category_name")["price"].median().sort_values(ascending=False).index
cat_data = [df[df["category_name"] == c]["price"].values for c in cats_sorted]
bp = ax.boxplot(cat_data, labels=cats_sorted, patch_artist=True, vert=True,
                medianprops=dict(color="#f0f6fc", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_title("Price Distribution by Category", fontsize=14, fontweight="bold")
ax.set_ylabel("Price ($)")
ax.set_xticklabels(cats_sorted, rotation=30, ha="right", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "07_price_by_category.png"), dpi=150)
plt.close()

# ================================================================
# 8. Daily Revenue Time Series with Anomaly Detection
# ================================================================
print("[8/10] Daily revenue & anomaly detection...")
daily_rev = df.groupby("sale_date")["sales_amount_realistic"].sum()
daily_rev.index = pd.to_datetime(daily_rev.index)
daily_rev = daily_rev.sort_index()

# Rolling mean and std for anomaly detection
roll_mean = daily_rev.rolling(30, center=True).mean()
roll_std  = daily_rev.rolling(30, center=True).std()
upper = roll_mean + 2.5 * roll_std
lower = roll_mean - 2.5 * roll_std
anomalies = daily_rev[(daily_rev > upper) | (daily_rev < lower)]

fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(daily_rev.index, daily_rev.values, color=PALETTE[0], alpha=0.4, linewidth=0.5)
ax.plot(roll_mean.index, roll_mean.values, color=PALETTE[1], linewidth=2, label="30-day rolling mean")
ax.fill_between(roll_mean.index, lower.values, upper.values, alpha=0.1, color=PALETTE[2])
if len(anomalies) > 0:
    ax.scatter(anomalies.index, anomalies.values, color=PALETTE[3], s=15, zorder=5, label=f"Anomalies ({len(anomalies)})")
ax.set_title("Daily Revenue with Anomaly Detection (2.5-sigma)", fontsize=14, fontweight="bold")
ax.set_ylabel("Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend(framealpha=0.3, edgecolor="#30363d")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "08_daily_revenue_anomalies.png"), dpi=150)
plt.close()
print(f"  Found {len(anomalies)} anomalous days")

# ================================================================
# 9. Product Recency Distribution Check
# ================================================================
print("[9/10] Product recency validation...")
fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
for i, year in enumerate(range(2021, 2026)):
    yd = df[df["year"] == year].copy()
    yd["launch_year"] = yd["launch_date"].apply(lambda x: pd.to_datetime(x).year)
    tier_counts = []
    labels = ["Current", "Y-1", "Y-2", "Older", "Subs"]
    tier_counts.append(len(yd[(yd["launch_year"] == year) & (yd["category_id"] != "CAT-8")]))
    tier_counts.append(len(yd[(yd["launch_year"] == year - 1) & (yd["category_id"] != "CAT-8")]))
    tier_counts.append(len(yd[(yd["launch_year"] == year - 2) & (yd["category_id"] != "CAT-8")]))
    tier_counts.append(len(yd[(yd["launch_year"] <= year - 3) & (yd["category_id"] != "CAT-8")]))
    tier_counts.append(len(yd[yd["category_id"] == "CAT-8"]))
    total = sum(tier_counts)
    pcts = [c / total * 100 for c in tier_counts]
    bars = axes[i].bar(labels, pcts, color=PALETTE[:5])
    axes[i].set_title(str(year), fontsize=12, fontweight="bold")
    axes[i].set_ylim(0, 75)
    for bar, pct in zip(bars, pcts):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{pct:.0f}%", ha="center", fontsize=8)
    if i == 0:
        axes[i].set_ylabel("% of Transactions")
    axes[i].tick_params(axis="x", rotation=30)
fig.suptitle("Product Recency Distribution Validation (Target: 60/20/10/5/5)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "09_recency_validation.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# 10. Summary Statistics Dashboard
# ================================================================
print("[10/10] Summary statistics...")
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# 10a: Revenue by year
ax1 = fig.add_subplot(gs[0, 0])
yr_rev = df.groupby("year")["sales_amount_realistic"].sum()
ax1.bar(yr_rev.index, yr_rev.values, color=PALETTE[0])
ax1.set_title("Total Revenue by Year", fontsize=11, fontweight="bold")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e9:.1f}B"))

# 10b: Avg transaction value by year
ax2 = fig.add_subplot(gs[0, 1])
yr_avg = df.groupby("year")["sales_amount_realistic"].mean()
ax2.bar(yr_avg.index, yr_avg.values, color=PALETTE[1])
ax2.set_title("Avg Transaction Value by Year", fontsize=11, fontweight="bold")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

# 10c: Products per year
ax3 = fig.add_subplot(gs[0, 2])
yr_prod = df.groupby("year")["product_name"].nunique()
ax3.bar(yr_prod.index, yr_prod.values, color=PALETTE[4])
ax3.set_title("Unique Products per Year", fontsize=11, fontweight="bold")

# 10d: Store distribution (top 10)
ax4 = fig.add_subplot(gs[1, :2])
top_stores = df.groupby("store_name")["sales_amount_realistic"].sum().nlargest(10)
ax4.barh(range(len(top_stores)), top_stores.values, color=PALETTE[2])
ax4.set_yticks(range(len(top_stores)))
ax4.set_yticklabels(top_stores.index, fontsize=8)
ax4.set_title("Top 10 Stores by Revenue", fontsize=11, fontweight="bold")
ax4.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.0f}M"))
ax4.invert_yaxis()

# 10e: Key stats text
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis("off")
stats_text = (
    f"DATASET SUMMARY\n"
    f"{'='*30}\n"
    f"Total Rows: {len(df):,}\n"
    f"Date Range: 2021-01-01\n"
    f"         to 2025-12-31\n"
    f"Products:   {df['product_name'].nunique()}\n"
    f"Stores:     {df['store_id'].nunique()}\n"
    f"Countries:  {df['country_norm_mapped'].nunique()}\n"
    f"\nTotal Revenue:\n"
    f"  ${df['sales_amount_realistic'].sum():,.0f}\n"
    f"Avg Daily Rev:\n"
    f"  ${df.groupby('sale_date')['sales_amount_realistic'].sum().mean():,.0f}\n"
    f"Median Qty: {df['quantity_realistic'].median():.0f}\n"
)
ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#21262d", edgecolor="#30363d"))

# 10f: Monthly seasonality (avg across all years)
ax6 = fig.add_subplot(gs[2, :])
month_avg = df.groupby("month")["sales_amount_realistic"].mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax6.bar(range(1, 13), month_avg.values, color=PALETTE[5])
ax6.set_xticks(range(1, 13))
ax6.set_xticklabels(month_names)
ax6.set_title("Average Sales Amount by Month (Seasonality)", fontsize=11, fontweight="bold")
ax6.set_ylabel("Avg Sales ($)")
ax6.grid(True, alpha=0.3, axis="y")

fig.suptitle("Apple Retail Sales - Comprehensive Dashboard", fontsize=16, fontweight="bold", y=1.01)
plt.savefig(os.path.join(OUT, "10_dashboard.png"), dpi=150, bbox_inches="tight")
plt.close()

# ================================================================
# Print Statistical Summary
# ================================================================
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(f"\n--- Date Range ---")
print(f"  Start: {df['sale_date'].min()}")
print(f"  End:   {df['sale_date'].max()}")
print(f"  Days:  {(pd.to_datetime(df['sale_date'].max()) - pd.to_datetime(df['sale_date'].min())).days}")

print(f"\n--- Volume ---")
print(f"  Total rows:      {len(df):,}")
print(f"  Unique products: {df['product_name'].nunique()}")
print(f"  Unique stores:   {df['store_id'].nunique()}")
print(f"  Unique countries:{df['country_norm_mapped'].nunique()}")

print(f"\n--- Revenue ---")
print(f"  Total:   ${df['sales_amount_realistic'].sum():,.2f}")
print(f"  Mean:    ${df['sales_amount_realistic'].mean():,.2f}")
print(f"  Median:  ${df['sales_amount_realistic'].median():,.2f}")
print(f"  Std:     ${df['sales_amount_realistic'].std():,.2f}")
print(f"  Min:     ${df['sales_amount_realistic'].min():,.2f}")
print(f"  Max:     ${df['sales_amount_realistic'].max():,.2f}")

print(f"\n--- Quantity (Realistic) ---")
print(f"  Mean:   {df['quantity_realistic'].mean():.2f}")
print(f"  Median: {df['quantity_realistic'].median():.1f}")
print(f"  Max:    {df['quantity_realistic'].max()}")
print(f"  Zeros:  {(df['quantity_realistic'] == 0).sum():,} ({(df['quantity_realistic']==0).mean()*100:.1f}%)")

print(f"\n--- Price (Realistic) ---")
print(f"  Mean:   ${df['price_realistic'].mean():,.2f}")
print(f"  Median: ${df['price_realistic'].median():,.2f}")
print(f"  Std:    ${df['price_realistic'].std():,.2f}")

print(f"\n--- Year Distribution ---")
for y, c in df.groupby("year").size().items():
    rev = df[df["year"]==y]["sales_amount_realistic"].sum()
    print(f"  {y}: {c:,} rows | ${rev:,.0f} revenue")

print(f"\n--- Month Distribution (avg across years) ---")
for m, c in df.groupby("month").size().items():
    print(f"  Month {m:2d}: {c:,} rows (avg {c//5:,}/year)")

print(f"\n--- Category Distribution ---")
for cat, grp in df.groupby("category_name"):
    cnt = len(grp)
    rev = grp["sales_amount_realistic"].sum()
    print(f"  {cat:25s}: {cnt:>8,} rows ({cnt/len(df)*100:4.1f}%) | ${rev:>15,.0f}")

print(f"\n--- Anomalies (Daily Revenue) ---")
print(f"  Anomalous days: {len(anomalies)}")
if len(anomalies) > 0:
    print(f"  Top anomalous dates:")
    for date, val in anomalies.nlargest(5).items():
        print(f"    {date.date()}: ${val:,.0f}")

print(f"\n--- Invalid Launch Flags ---")
invalid = df["invalid_launch_flag"].sum()
print(f"  Rows with future launch date: {invalid:,} ({invalid/len(df)*100:.2f}%)")

print(f"\n--- Product Age (days) ---")
print(f"  Mean:   {df['product_age_days'].mean():,.0f}")
print(f"  Median: {df['product_age_days'].median():,.0f}")
print(f"  Max:    {df['product_age_days'].max():,}")
print(f"  Negative (invalid): {(df['product_age_days'] < 0).sum():,}")

print(f"\nPlots saved to: {OUT}")
print("[DONE] Analysis complete!")
