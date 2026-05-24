# Power BI Dashboard Blueprint: Apple Sales Intelligence (V3)

> **Data Version:** V3 — `cleaned_apple_sales_v3.csv`  
> **Date Range:** January 2021 – December 2025  
> **Markets:** 19 countries · 75 stores · 89 products  
> **Total Transactions:** ~1,068,918 records  

Based on the **Star Schema** data model we built—`Fact_Sales`, `Dim_Product`, `Dim_Store`, and `Dim_Macroeconomics`—here is the recommended blueprint for your Power BI report.

By separating the analysis into these **5 distinct pages**, each page tells a unique story without overwhelming the viewer.

---

## Data Model Relationships (Model View)

| Relationship | Type | Join Key(s) |
|---|---|---|
| `Dim_Product` → `Fact_Sales` | 1 : Many | `product_id` |
| `Dim_Store` → `Fact_Sales` | 1 : Many | `store_id` |
| `Dim_Calendar` → `Fact_Sales` | 1 : Many | `Date` ↔ `sale_date` |
| `Dim_Macroeconomics` → `Fact_Sales` | 1 : Many | `CountryYear_Key` (composite) |

### Key DAX Measures to Create

```dax
Total Revenue = SUM(Fact_Sales[sales_amount_realistic])

Total Units = SUM(Fact_Sales[quantity_realistic])

Avg Order Value = DIVIDE([Total Revenue], COUNTROWS(Fact_Sales))

Promo Rate = DIVIDE(
    CALCULATE(COUNTROWS(Fact_Sales), Fact_Sales[promo_flag] = 1),
    COUNTROWS(Fact_Sales)
)

YoY Growth = 
VAR CurrentYear = [Total Revenue]
VAR PrevYear = CALCULATE([Total Revenue], DATEADD(Dim_Calendar[Date], -1, YEAR))
RETURN DIVIDE(CurrentYear - PrevYear, PrevYear)
```

---

## Page 1: Executive Overview
**The "Bird's-Eye View" for leadership to instantly gauge business health.**

* **KPI Cards (Top row, 6 cards):**
  * Total Revenue (`sales_amount_realistic`) — formatted as currency
  * Total Units Sold (`quantity_realistic`) — whole number
  * Total Transactions — `COUNTROWS(Fact_Sales)`
  * Avg Order Value — `[Total Revenue] / COUNTROWS(Fact_Sales)`
  * Promo Rate — `% of transactions where promo_flag = 1`
  * Active Markets — `DISTINCTCOUNT(Dim_Store[country_norm_mapped])`
* **Area Chart (Main Visual):**
  * *Axis:* `Dim_Calendar[Date]` (Year/Month hierarchy)
  * *Values:* `[Total Revenue]`
  * *Insight:* Shows the revenue trajectory across 2021–2025.
* **Donut Chart:**
  * *Legend:* `Dim_Product[category_name]`
  * *Values:* `[Total Revenue]`
  * *Insight:* Which categories (Smartphone, Laptop, Wearable, Audio, Accessories, Subscription Service, Desktop, Software, Tablet) drive the most revenue.
* **Decomposition Tree:**
  * *Analyze:* `[Total Revenue]` → *Explain By:* `country_norm_mapped` then `category_name`
  * *Insight:* Interactively drill down into the highest performing regions and what they buy.

---

## Page 2: Product & Launch Analytics
**Investigates how product age, trends, and launches dictate success across 89 products.**

* **Clustered Bar Chart:**
  * Top 10 products by `[Total Revenue]`.
* **Scatter Plot:**
  * *X-Axis:* `product_age_days` (or `days_from_start`)
  * *Y-Axis:* `sales_amount_realistic`
  * *Legend:* `category_name`
  * *Insight:* Uncovers the "decay curve" — do sales spike at launch and flatline, or hold steady?
* **Ribbon / Line Chart:**
  * *Axis:* `Dim_Calendar[Month]`
  * *Values:* `[Total Units]` split by `product_name` (filter to top 5).
  * *Insight:* Visualizes how top products compete against each other over time.
* **Table / Matrix:**
  * Highlighting rows where `invalid_launch_flag` = True.
  * *Columns:* Product Name, Category, Launch Date, Total Revenue, Units Sold.
  * *Insight:* Data quality check — does flagged launch data mask high revenue?
* **Pie Chart:**
  * Revenue share by `category_name`.
  * *Insight:* Quick visual of category dominance.

---

## Page 3: Geographic & Store Optimization
**For regional managers — 19 markets, 75 stores.**

* **Filled Map:**
  * *Location:* `country_norm_mapped`
  * *Color Saturation:* `[Total Revenue]`
  * *Insight:* Identifies the best-performing markets globally at a glance.
* **Horizontal Bar Chart:**
  * Top 20 Stores by `[Total Revenue]`, colored by Country.
* **Matrix Visual:**
  * *Rows:* Country → City → Store Name
  * *Columns:* Year → Quarter
  * *Values:* `[Total Revenue]` with conditional data bars.
  * *Insight:* Deep operational drill-down for local auditing.
* **Bar Chart:**
  * Average Revenue per Store by Country = `[Total Revenue] / DISTINCTCOUNT(store_id)`.
  * *Insight:* Normalizes for countries with many vs few stores.
* **Heatmap (Matrix with conditional formatting):**
  * Top 20 Stores × Year — `[Total Revenue]` with color scale.
  * *Insight:* Spot year-over-year performance shifts per store.

---

## Page 4: Promotions & Seasonality (The "Lift" Page)
**Determines if marketing budgets actually generated ROI using `promo_flag` and `season_factor`.**

* **Clustered Column Chart:**
  * *Axis:* `Dim_Calendar[Month Name]`
  * *Values:* `[Total Revenue]`
  * *Legend:* `promo_flag` (Promo / No Promo)
  * *Insight:* Visually confirms if months with active promos outperformed other months.
* **Line and Stacked Column Chart:**
  * *Column Values:* `[Total Revenue]`
  * *Line Value:* Average `season_factor`
  * *Axis:* Month
  * *Insight:* Proves if sales spikes are due to holidays/seasonality or underlying organic trend.
* **Gauge Charts (×3):**
  * 1) `SUM(quantity)` vs `SUM(quantity_realistic)` — baseline vs modelled units.
  * 2) `SUM(sales_amount)` vs `SUM(sales_amount_realistic)` — baseline vs modelled revenue.
  * 3) Average `promo_factor` — how much lift promo provides.
  * *Insight:* Shows the variance between baseline and enriched/adjusted metrics.
* **Bar Chart:**
  * Avg Transaction Value: Promo vs No-Promo, broken down by `category_name`.
  * *Insight:* Which categories benefit most from promotions?

---

## Page 5: Macroeconomic Context & Externalities
**Uses `Dim_Macroeconomics` (joined via `CountryYear_Key`) to explain how outside forces impact Apple's bottom line.**

* **Scatter Plot with Play Axis (Animation):**
  * *X-Axis:* `gdp_per_capita`
  * *Y-Axis:* `[Total Revenue]`
  * *Size:* `internet_usage_pct`
  * *Play Axis:* `Dim_Calendar[Year]`
  * *Insight:* Watch countries evolve over 2021–2025 — does GDP growth track with Apple sales?
* **Dual-Axis Line Chart:**
  * *Axis:* Year
  * *Line 1:* `[Total Revenue]`
  * *Line 2:* Average `inflation_rate`
  * *Insight:* Does high inflation correlate with declining realistic sales?
* **Bar Charts (side by side):**
  * GDP per Capita by Country (sorted descending).
  * Avg Inflation Rate by Country (sorted descending).
  * *Insight:* Quick visual ranking of macro-indicators.
* **Line Charts (side by side):**
  * Exchange Rate vs USD by Year (multi-country selection).
  * Inflation Rate by Year (multi-country selection).
  * *Insight:* Track currency and price stability trends for key markets.
* **Bar Chart:**
  * Average Exchange Rate vs USD by Country.
  * *Insight:* Identifies markets with the most volatile or extreme currency situations.

---

## Power BI Best Practices Checklist

- [ ] Create a **Date Table** using `CALENDARAUTO()` and mark it as a Date Table.
- [ ] Create the **`CountryYear_Key`** composite key in both `Dim_Macroeconomics` and `Fact_Sales` for the macro join.
- [ ] **Hide** foreign key columns (`store_id`, `product_id`, `category_id`) from Report View.
- [ ] Set all relationships to **single-direction cross-filter** unless specifically needed.
- [ ] Use **Import mode** (not DirectQuery) for the CSV files.
- [ ] Format all currency measures with `$#,##0` or `$#,##0.00`.
- [ ] Add **bookmarks** for a "Reset Filters" button on each page.
- [ ] Enable **Row-level security (RLS)** if sharing with regional managers (filter by `country_norm_mapped`).
