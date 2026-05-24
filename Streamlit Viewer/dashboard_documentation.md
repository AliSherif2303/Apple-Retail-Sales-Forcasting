# Apple Sales Intelligence Dashboard — Technical Documentation & KPI Analysis

> **Author:** Data Analytics Team  
> **Dashboard Tool:** Streamlit + Plotly  
> **Data Version:** V3 — `cleaned_apple_sales_v3.csv`  
> **Date Range:** January 2021 – December 2025  
> **Last Updated:** April 2026  

---

## 1. Architecture & Data Pipeline

### 1.1 Data Model (Star Schema)

The dashboard consumes a **Star Schema** consisting of one Fact table and three Dimension tables, all derived from a single enriched CSV file (`cleaned_apple_sales_v3.csv`, ~223 MB).

```
                    ┌──────────────────┐
                    │  Dim_Product     │
                    │  (89 products)   │
                    └────────┬─────────┘
                             │ product_id
┌──────────────────┐         │         ┌──────────────────────┐
│   Dim_Store      │─────────┼─────────│  Dim_Macroeconomics  │
│   (75 stores)    │ store_id│         │  (19 countries × 5yr)│
└──────────────────┘         │         └──────────────────────┘
                             │ year + country_norm_mapped
                    ┌────────┴─────────┐
                    │   Fact_Sales     │
                    │  (1,068,918 txns)│
                    └──────────────────┘
```

### 1.2 Table Specifications

| Table | Rows | Columns | Size | Primary Key |
|-------|------|---------|------|-------------|
| `Fact_Sales.csv` | 1,068,918 | 24 | 223 MB | `sale_id` |
| `Dim_Product.csv` | 89 | 6 | 7 KB | `product_id` |
| `Dim_Store.csv` | 75 | 4 | 3 KB | `store_id` |
| `Dim_Macroeconomics.csv` | 95 | 7 | 7 KB | `country_norm_mapped` + `year` |

### 1.3 Column Dictionary

#### Fact_Sales

| Column | Type | Description |
|--------|------|-------------|
| `sale_id` | string | Unique transaction identifier |
| `sale_date` | date | Transaction date (2021-01-01 to 2025-12-31) |
| `store_id` | string | FK → Dim_Store |
| `product_id` | string | FK → Dim_Product |
| `quantity` | int | Baseline units sold (pre-enrichment) |
| `price` | int | Baseline unit price (pre-enrichment) |
| `sales_amount` | int | Baseline revenue = quantity × price |
| `quantity_realistic` | int | Modelled units (adjusted for seasonality, economy, promotions) |
| `price_realistic` | float | Modelled price (adjusted for exchange rate, inflation) |
| `sales_amount_realistic` | float | **Primary revenue metric** = quantity_realistic × price_realistic |
| `promo_flag` | int (0/1) | Whether a promotion was active during this transaction |
| `promo_factor` | float | Multiplier applied when promo_flag = 1 (typically 1.0–1.4) |
| `season_factor` | float | Seasonal demand multiplier (0.8 = low season, 1.2 = peak) |
| `economic_factor` | float | Country-level economic adjustment factor |
| `trend_factor` | float | Product lifecycle trend multiplier |
| `store_factor` | float | Store-level performance multiplier |
| `mu_demand` | float | Modelled expected demand (Poisson λ parameter) |
| `product_age_days` | int | Days since product launch at time of sale |
| `days_from_start` | int | Days since the start of the dataset (2021-01-01) |
| `invalid_launch_flag` | bool | True if the product's launch_date is questionable |
| `category_id` | string | FK → category (denormalized into Dim_Product) |
| `country_norm_mapped` | string | FK → Dim_Store / Dim_Macroeconomics |
| `year` | int | Sale year (2021–2025) |
| `month` | int | Sale month (1–12) |

#### Dim_Product

| Column | Type | Description |
|--------|------|-------------|
| `product_id` | string | Primary key (P-1 to P-98) |
| `product_name` | string | e.g., "iPhone 17 Pro", "MacBook Air M3" |
| `category_id` | string | Category identifier |
| `category_name` | string | One of 10 categories (Smartphone, Laptop, Desktop, Tablet, Wearable, Audio, Accessories, Streaming Device, Subscription Service, Smart Speaker) |
| `launch_date` | date | Product launch date |
| `product_trend` | float | Long-term demand trend coefficient |

#### Dim_Store

| Column | Type | Description |
|--------|------|-------------|
| `store_id` | string | Primary key (ST-1 to ST-75) |
| `store_name` | string | e.g., "Apple Fifth Avenue" |
| `city` | string | Store city |
| `country_norm_mapped` | string | Normalized country name (lowercase) |

#### Dim_Macroeconomics

| Column | Type | Description |
|--------|------|-------------|
| `country_norm_mapped` | string | Composite PK (part 1) |
| `year` | int | Composite PK (part 2) |
| `exchange_rate` | float | Local currency per 1 USD |
| `inflation_rate` | float | Annual inflation rate (%) |
| `internet_usage_pct` | float | Internet penetration (%) |
| `gdp_type` | string | GDP metric type (GDP_per_capita or GDP_total) |
| `gdp_per_capita` | float | GDP per capita in USD |

---

## 2. Dashboard Pages & Visual Inventory

### 2.1 Tab 1 — Sales Trends

| Visual | Chart Type | X-Axis | Y-Axis | Filters |
|--------|-----------|--------|--------|---------|
| Sales Revenue Trend | Area Chart | `yearmonth` / quarter / year | `sales_amount_realistic` | Granularity selector |
| Revenue by Category & Year | Grouped Bar | `year` | `sales_amount_realistic` | Colored by `category_name` |
| Monthly Seasonality | Bar Chart | `month_name` | Avg `sales_amount_realistic` | Color scale by value |
| Promo vs Non-Promo Revenue | Dual Line | `yearmonth` | `sales_amount_realistic` | Split by `promo_flag` |

### 2.2 Tab 2 — Market Analysis

| Visual | Chart Type | X-Axis | Y-Axis | Filters |
|--------|-----------|--------|--------|---------|
| Revenue by Country | Horizontal Bar | Revenue | Country | Color scale |
| Market Share Treemap | Treemap | — | — | Sized by revenue |
| Top-10 Country Revenue Trend | Multi-line | Year | Revenue | Top 10 countries |
| Units Sold by Country | Bar | Country | `quantity` | Top 15 |
| Avg Order Value by Country | Bar | Country | AOV | Top 15 |

### 2.3 Tab 3 — Product Insights

| Visual | Chart Type | X-Axis | Y-Axis | Filters |
|--------|-----------|--------|--------|---------|
| Revenue by Category | Donut / Pie | — | — | `category_name` |
| Units Sold by Category | Bar | Category | `quantity` | — |
| Top N Products | Horizontal Bar | Revenue | `product_name` | Slider: 5–30 |
| Price Distribution | Box Plot | `category_name` | `price` | — |
| Quantity Distribution | Histogram | `quantity` | Count | — |
| Promo Impact by Category | Grouped Bar | Category | Avg `sales_amount_realistic` | Promo vs No Promo |

### 2.4 Tab 4 — Store Performance

| Visual | Chart Type | X-Axis | Y-Axis | Filters |
|--------|-----------|--------|--------|---------|
| Top 20 Stores | Horizontal Bar | Revenue | Store Name | Colored by country |
| Store Count by Country | Bar | Country | Count | — |
| Revenue per Store by Country | Bar | Country | Rev / Store | — |
| Revenue Heatmap | Heatmap (imshow) | Year | Store Name (Top 20) | Color = Revenue |

### 2.5 Tab 5 — Economic Factors

| Visual | Chart Type | X-Axis | Y-Axis | Filters |
|--------|-----------|--------|--------|---------|
| GDP per Capita by Country | Bar | Country | Avg GDP | Color scale |
| Avg Inflation by Country | Bar | Country | Avg Inflation | Color scale |
| GDP vs Revenue Scatter | Bubble Scatter | GDP per Capita | Revenue | Size = Revenue |
| Exchange Rate by Year | Multi-line | Year | FX Rate | Country multi-select |
| Inflation Rate by Year | Multi-line | Year | Inflation | Country multi-select |
| Avg Exchange Rate by Country | Bar | Country | Avg FX | Color scale |

### 2.6 Global Sidebar Filters

| Filter | Type | Default |
|--------|------|---------|
| Year(s) | Multi-select | All (2021–2025) |
| Country/Market | Multi-select | All 19 countries |
| Product Category | Multi-select | All 10 categories |
| Promotion Filter | Radio | "All" / "Promo Only" / "No Promo" |

---

## 3. KPI Analysis

### 3.1 Executive KPIs

| KPI | Value | Analysis |
|-----|-------|----------|
| **Total Revenue** | **$5.22 B** | Realistic-adjusted revenue across 5 years and 19 markets. This is the primary metric for all revenue visualizations. |
| **Total Units Sold** | **8,299,238** | Modelled quantity after applying seasonal, economic, and promotional adjustments. |
| **Total Transactions** | **1,068,918** | One row per sale event. Average basket size = 7.8 units per transaction. |
| **Avg Order Value** | **$4,882** | High AOV driven by premium product mix (iPhones, MacBooks). Indicates Apple's pricing power. |
| **Promo Rate** | **15.8%** | Only ~1 in 6 transactions occur during a promotion window — promotions are targeted, not blanket. |
| **Active Markets** | **19** | Global footprint spanning North America, Europe, Asia-Pacific, Middle East, and Latin America. |

### 3.2 Revenue Growth Trajectory

| Year | Revenue | YoY Growth |
|------|---------|------------|
| 2021 | $717.4 M | — (baseline) |
| 2022 | $801.3 M | **+11.7%** |
| 2023 | $1.04 B | **+30.2%** |
| 2024 | $1.19 B | **+14.2%** |
| 2025 | $1.47 B | **+23.1%** |

**Analyst Insight:** Revenue has more than doubled from 2021 to 2025, growing from $717M to $1.47B. The 30.2% spike in 2023 is the most aggressive growth year. This likely correlates with the combination of post-pandemic recovery, new iPhone cycle launches (iPhone 15/16 era), and expansion of the product catalog. Growth decelerated slightly in 2024 (+14.2%) before re-accelerating in 2025 (+23.1%), suggesting a strong new product cycle (iPhone 17 series) drove renewed momentum.

### 3.3 Category Revenue Breakdown

| Category | Revenue | Share | Insight |
|----------|---------|-------|---------|
| **Smartphone** | $2.81 B | **53.8%** | Over half of all revenue. iPhone remains the undisputed cash cow. |
| **Laptop** | $647.4 M | 12.4% | MacBook line is the second pillar. |
| **Desktop** | $556.6 M | 10.7% | iMac/Mac Pro — strong enterprise/creative segment. |
| **Tablet** | $549.5 M | 10.5% | iPad line nearly matches Desktop revenue. |
| **Wearable** | $455.1 M | 8.7% | Apple Watch + accessories — growing segment. |
| **Audio** | $154.8 M | 3.0% | AirPods line. Relatively small but high-margin. |
| **Accessories** | $18.1 M | 0.3% | Cases, cables, chargers — low ASP. |
| **Streaming Device** | $17.4 M | 0.3% | Apple TV — niche. |
| **Subscription Service** | $10.8 M | 0.2% | Apple One, Music, TV+ — low per-txn value but recurring. |
| **Smart Speaker** | $3.1 M | 0.1% | HomePod — smallest category by far. |

**Analyst Insight:** The **Smartphone category alone generates 53.8% of total revenue**, making it the single biggest risk and opportunity. A diversification strategy into Wearables and Services would reduce dependency. Notably, the combined "ecosystem" categories (Audio + Wearable + Subscription) represent 11.9% and are likely growing faster than hardware.

### 3.4 Geographic Performance

#### Top 5 Markets by Revenue

| Rank | Country | Revenue | Share |
|------|---------|---------|-------|
| 1 | **United States** | $1.06 B | **20.3%** |
| 2 | Australia | $487.3 M | 9.3% |
| 3 | China | $487.1 M | 9.3% |
| 4 | Japan | $435.0 M | 8.3% |
| 5 | Canada | $364.4 M | 7.0% |

**Analyst Insight:** The US accounts for ~20% of global revenue, which is a healthy lead without dangerous over-concentration. Interestingly, **Australia and China are virtually tied** at $487M each — suggesting Australia punches well above its population weight (26M vs China's 1.4B). This implies a much higher per-capita Apple spend in Australia, likely driven by high GDP ($65K) and internet penetration (97%).

#### Top 5 Stores

| Store | City | Revenue |
|-------|------|---------|
| Apple Central World | Bangkok | $154.3 M |
| Apple Covent Garden | London | $153.2 M |
| Apple Champs-Élysées | Paris | $134.4 M |
| Apple The Dubai Mall | Dubai | $117.6 M |
| Apple Orchard Road | Singapore | $106.6 M |

**Analyst Insight:** The top-performing stores are predominantly in **tourist-heavy, high-traffic global cities** rather than in the US (which dominates at the country level via volume across many stores). Bangkok's Apple Central World leads — likely driven by favorable exchange rates, tourism traffic, and a high `store_factor` multiplier.

### 3.5 Promotion Effectiveness

| Metric | Promo (flag=1) | No Promo (flag=0) | Lift |
|--------|---------------|-------------------|------|
| Avg Transaction Value | **$5,889** | $4,692 | **+25.5%** |
| % of Transactions | 15.8% | 84.2% | — |

**Analyst Insight:** Promotions deliver a clear **+25.5% lift** in average transaction value. Despite only 15.8% of transactions being promotional, they disproportionately drive revenue. However, this needs nuance — the `promo_factor` in the model is typically set to 1.4× for promo transactions, meaning the enrichment model *assumes* a 40% quantity boost. The actual observed lift of +25.5% in dollar terms (after price adjustments) suggests the model's assumptions are reasonable but conservative.

### 3.6 Seasonality Pattern

| Month | Avg Monthly Revenue | Index vs. Mean |
|-------|-------------------|----------------|
| Jan | $69.0 M | 79% |
| Feb | $57.2 M | **65% (lowest)** |
| Mar | $75.1 M | 86% |
| Apr | $71.8 M | 82% |
| May | $73.0 M | 83% |
| Jun | $74.4 M | 85% |
| Jul | $76.1 M | 87% |
| Aug | $80.4 M | 92% |
| Sep | $108.0 M | 123% |
| Oct | $104.4 M | 119% |
| Nov | $119.9 M | 137% |
| Dec | $134.5 M | **153% (highest)** |

**Analyst Insight:** There is a **dramatic seasonal ramp** starting in September and peaking in December. December revenue ($134.5M/month avg) is **2.35× February** ($57.2M). This aligns perfectly with Apple's real-world product launch cycle (new iPhones in September) and holiday gift-giving season (Nov–Dec). The `season_factor` in the model captures this well, with values rising from 0.8 in Q1 to 1.2 in Q4.

### 3.7 Macroeconomic Correlations

| Factor | Correlation with Revenue | Interpretation |
|--------|------------------------|----------------|
| GDP per Capita | **+0.32** (weak positive) | Wealthier countries tend to buy more Apple products, but it's not a strong predictor alone. |
| Inflation Rate | **−0.11** (negligible negative) | Inflation has minimal direct impact on Apple sales — brand loyalty and pricing power absorb inflation. |
| Internet Usage % | **+0.05** (negligible) | At 70–100% internet penetration across all 19 markets, this variable has near-zero variance to explain. |

**Analyst Insight:** The weak GDP correlation (+0.32) is notable because it suggests Apple's revenue is **not purely a function of national wealth**. Markets like Thailand and Colombia — with lower GDP — still generate significant revenue, likely driven by aspirational purchasing behavior and tourism. The near-zero inflation correlation confirms Apple's exceptional **pricing power** — consumers continue to buy even when inflation rises.

### 3.8 Baseline vs. Realistic (Model Validation)

| Metric | Baseline (raw) | Realistic (modelled) | Delta |
|--------|---------------|---------------------|-------|
| Total Quantity | 5,879,596 | 8,299,238 | +41.2% |
| Total Revenue | $5.93 B | $5.22 B | −11.9% |

**Analyst Insight:** The enrichment model **increased quantity by 41%** (due to promotional and seasonal multipliers boosting unit sales) but **decreased revenue by 12%** (due to realistic price adjustments accounting for exchange rates, inflation, and economic factors). This produces a more conservative revenue figure than the naive baseline — which is desirable for planning purposes. The model essentially says: "More units move, but at lower real prices."

---

## 4. Technical Notes

### 4.1 Memory Optimization
The dashboard applies three memory optimization techniques during data loading:
- **Category encoding:** Low-cardinality string columns are converted to `pandas.Categorical` (saves ~60% memory on string columns).
- **Float downcasting:** `float64` → `float32` (halves float memory).
- **Integer downcasting:** `int64` → smallest fitting int type via `pd.to_numeric(downcast='integer')`.

### 4.2 Caching
All data loading is wrapped in `@st.cache_data`, so the CSV files are read and merged only once per session. Subsequent filter changes use the in-memory DataFrame.

### 4.3 Known Limitations
1. **No real-time data feed** — the dashboard reads static CSV files. Updates require re-running the split script and redeploying.
2. **Dim_Macroeconomics joins at year granularity** — economic indicators don't vary monthly, so sub-annual macro analysis is limited.
3. **`invalid_launch_flag`** — some products have questionable launch dates. These are included in totals but should be flagged in any product-level audit.
4. **Git LFS required** — `Fact_Sales.csv` is 223 MB and requires Git LFS for GitHub hosting.

### 4.4 Deployment
- **Local:** `python -m streamlit run dashboard.py`
- **Streamlit Cloud:** Deployed via GitHub repo (`https://github.com/ahmed-adel911s/Streamlit`). Auto-deploys on push to `main`.
- **Requirements:** `streamlit>=1.32.0`, `pandas>=2.0.0`, `plotly>=5.18.0`, `openpyxl>=3.1.0`
