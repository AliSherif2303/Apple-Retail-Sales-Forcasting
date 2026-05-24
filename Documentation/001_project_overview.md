# 🍏 Apple Retail Sales Forecasting — Full Project Explanation

> **Project:** Apple Retail Sales Forecasting  
> **Authors:** Ali Sherif Salaheldin, Ali Mohamed, Hassan saad, Ahmed adel, Mohammed azzam  
> **GitHub:** [AliSherif2303/Apple-Retail-Sales-Forcasting](https://github.com/AliSherif2303/Apple-Retail-Sales-Forcasting)

---

## 📌 Project Overview

This project builds an end-to-end **retail sales forecasting pipeline** for Apple's global store network. It:

1. Ingests & cleans raw Kaggle data (sales, products, stores, warranty, macro-economics)
2. Enriches and simulates realistic demand signals (seasonality, economic factors, promotions, product trends)
3. Trains time-series forecasting models (CatBoost, AdaBoost)
4. Performs market expansion analysis to identify the next best locations for Apple Stores
5. Hosts a Streamlit dashboard + SQL RAG agent (Ollama)

---

## 📁 Directory Structure

```
Apple-Retail-Sales-Forcasting/
│
├── data/
│   ├── raw/                         ← Original Kaggle CSVs
│   └── processed/                   ← Cleaned, enriched, versioned datasets
│
├── notebooks/                       ← Jupyter notebooks (EDA, models, expansion)
├── scripts/                         ← Python utility & pipeline scripts
├── Streamlit Viewer/                ← Streamlit dashboard application
├── Documentation/                   ← Markdown reports & strategy docs
├── reports/figures/                 ← Generated chart images
├── requirements.txt
└── README.md
```

---

## 🗄️ DATA LAYER

### Raw Data (`data/raw/`)

| File | Description | Shape |
|------|-------------|-------|
| `sales.csv` | 1,040,200 transactions (sale_id, store_id, product_id, quantity, sale_date) | ~1M rows |
| `products.csv` | Product catalog with IDs, categories, launch dates, prices | 89 rows |
| `stores.csv` | 75 Apple Stores — name, city, country | 75 rows |
| `category.csv` | 10 product categories (CAT-1 to CAT-10) | 10 rows |
| `warranty.csv` | 30,000 warranty claims & repair statuses | 30K rows |
| `gdp.csv` | GDP per capita by country/year | External |
| `infilation.csv` | Annual inflation rates by country | External |
| `exchange.csv` | Currency exchange rates | External |
| `Individuals using the Internet.csv` | Internet penetration % by country | External |
| `Year-Category-Product-StartingPrice.csv` | Historical Apple product pricing | Reference |

### Processed Data (`data/processed/`)

The pipeline produces versioned datasets with increasing realism:

| File | Description |
|------|-------------|
| `cleaned_apple_sales.csv` | Base merged & cleaned dataset (~140 MB) |
| `cleaned_apple_sales_enriched.csv` | + GDP, inflation, exchange rate, internet % (~208 MB) |
| `cleaned_apple_sales_enriched_realistic.csv` | + First-pass simulation factors (~356 MB) |
| `cleaned_apple_sales_v2.csv` | + Real Apple product redistribution & date shift to 2021-2025 (~366 MB) |
| `cleaned_apple_sales_v3.csv` | + Fixed simulation factors (final production dataset, ~379 MB) |
| `generated_products.csv` | 98 real Apple products (P-1 → P-98) with launch dates & prices |
| `merged_city_sales_data.csv` | City-level aggregated sales with population data (~384 MB) |
| `merged_apple_enriched_auto.csv` | Auto-merged enriched data (~203 MB) |
| `cleaned_exchange.csv` | Cleaned exchange rates |
| `cleaned_gdp.csv` | Cleaned GDP data |
| `cleaned_inflation.csv` | Cleaned inflation data |
| `cleaned_internet.csv` | Cleaned internet usage data |

---

## 📓 NOTEBOOKS

### Core Pipeline Notebooks

#### `notebooks/003_preprocess_external_macro_data.ipynb`
- First pass at cleaning external macro data (GDP, inflation, exchange rates, internet usage)
- Normalizes country names for joining

#### `notebooks/005_simulate_demand_factors.ipynb` ⭐ (Main Simulation Notebook)
- **Cell 31** is the core simulation cell that builds 12 demand-simulation columns:
  - `season_factor` — monthly multiplier
  - `economic_factor` — GDP/inflation-weighted demand
  - `promo_flag` / `promo_factor` — promotional events
  - `price_realistic` — inflation + promo-adjusted price
  - `product_trend` / `trend_factor` — lifecycle growth/decline
  - `store_factor` — per-store performance coefficient
  - `mu_demand` — combined demand signal
  - `quantity_realistic` — Negative Binomial draw from mu_demand
  - `sales_amount_realistic` — final revenue column

#### `notebooks/004_merge_macro_and_sales.ipynb`
- Merges the base sales data with all external macro datasets
- Performs country-name normalization

#### `notebooks/006_final_preprocessing_features.ipynb`
- Final preprocessing steps before modeling

#### `notebooks/001_data_loading_and_merging.ipynb`
- Initial EDA on raw data
- Merges products, stores, categories into sales

#### `notebooks/002_exploratory_data_analysis.ipynb`
- Full EDA with 2M+ record dataset
- Seasonal patterns, store performance, category trends

#### `notebooks/014_exploratory_data_analysis_enriched.ipynb`
- EDA on the enriched/realistic dataset

### Forecasting Model Notebooks

#### `notebooks/011_catboost_forecasting.ipynb` (generated via `scripts/generate_catboost_notebook.py`)
**CatBoost Time-Series Forecasting Pipeline:**
1. Loads `cleaned_apple_sales_enriched_realistic.csv`
2. Aggregates transactions → monthly store-level summaries (~3,600 rows)
3. Feature engineering:
   - Lag features: 1–6 month lags + 12-month lag for sales & quantity
   - Rolling stats: 3/6/12-month rolling mean, std, min, max
   - Cyclical encoding: sine/cosine of month & quarter
   - Economic rates of change (GDP Δ, inflation Δ, FX Δ, internet Δ)
   - Store & country target/label encoding
   - Trend & momentum features (MoM %, expanding mean)
4. 52 engineered features total
5. Train/test split on temporal hold-out (last 5 months)
6. Baseline CatBoost → GridSearchCV with TimeSeriesSplit (5 folds)
7. Evaluation: MAE, RMSE, MAPE, R²
8. Future forecast: Jan 2025 for all 75 stores

#### `notebooks/009_adaboost_forecasting.ipynb`
- AdaBoost Regressor alternative to CatBoost
- Same feature engineering pipeline

#### `notebooks/013_combined_ada_and_catboost.ipynb`
- Side-by-side comparison of CatBoost vs AdaBoost

#### `notebooks/cat&ada_regularized.ipynb` (Deleted)
- Regularized versions of both models (L1/L2)

#### `notebooks/007_preprocessing_merged_enriched_archive.ipynb`
- Large preprocessing notebook on the merged enriched dataset

#### `notebooks/008_date_shifting_utility.ipynb`
- Date shifting utility (moves data from 2020-2024 → 2021-2025)

#### `notebooks/015_ollama_sql_rag_agent.ipynb`
- SQL RAG agent using Ollama (local LLM)
- Queries `apple_sales_rag_ollama.db` (SQLite)
- Natural language → SQL → answer pipeline

### Market Expansion & Validation

#### `notebooks/016_market_expansion_store_clustering.ipynb`
- K-Means clustering and store expansion priority analysis

#### `notebooks/017_target_market_validator.py`
**Market Entry Opportunity & Sales Estimation System:**
1. Loads historical store transactions and country-level demographics (`merged_city_sales_data.csv`, `merged_data_with_population.csv`).
2. Runs a **Weighted Scoring System** across potential new markets:
   - GDP per capita (35%), Population (20%), Internet Penetration (15%), Inflation Rate stability (15%), Exchange Rate stability (5%), and similarity to successful historical markets (10%).
3. Classifies markets into Priority levels (Very High, High, Medium, Low, Very Low).
4. Estimates **Year-1 and Year-3 sales potential** using dynamic similarity metrics and risk penalties (adjusting for inflation, currency, and digital reach).
5. Suggests concrete entry strategies (e.g. EXPAND NOW, PILOT FIRST, MONITOR, AVOID).
6. Exports comprehensive intelligence data (`market_expansion_intelligence.csv`, `executive_summary_top_markets.csv`) and saves analytical plots.

---

## ⚙️ SCRIPTS

### Core Data Pipeline Scripts

#### `scripts/redistribute_products.py` ⭐
**Product Redistribution & Date Extension (creates v2):**
1. Loads `cleaned_apple_sales_enriched_realistic.csv`
2. Shifts all dates +1 year (2020-2024 → 2021-2025)
3. Generates new rows for Nov 13 – Dec 31, 2025 (~35,700 rows)
4. **Recency-Weighted Product Assignment:**
   - 60% current-year products (e.g., iPhone 16 in 2024)
   - 20% Y-1 products
   - 10% Y-2 products
   - 5% older products
   - 5% subscriptions
5. Recalculates all derived columns (sales_amount, product_age_days, price_realistic, etc.)
6. Saves as `cleaned_apple_sales_v2.csv` (~1,068,918 rows)

#### `scripts/fix_simulation_factors.py` ⭐
**Simulation Fix Pipeline (v2 → v3) — 4 Phases:**

**Phase 1 — Independent Factors:**
- `season_factor`: 12 distinct monthly multipliers based on Apple's real retail calendar
  - Sep (1.25, iPhone launch), Dec (1.45, holiday), Feb (0.80, slowest)
- `economic_factor`: Per-country normalization instead of global mean
- `product_trend`: Category-aware lifecycle trends (Wearables +0.0004/day, Smart Speakers -0.0003/day)
- `store_factor`: Annual drift ±3% per store per year

**Phase 2 — Dependent Factors:**
- `promo_flag`/`promo_factor`: 3-layer model (monthly base + age boost + category boost)
  - Nov: 25%, Aug: 15%, Jan: 5% base rates
  - Older products: +8-15% promo probability
- `price_realistic`: 4-layer model (inflation + age depreciation + promo discount + noise)
  - 3+ year old products: -20% depreciation

**Phase 3 — Demand Generation:**
- `mu_demand`: Category-aware (iPhones 1.4x, Mac Pro 0.5x) × price-inverse scaling
- Price elasticity: -1.2 for < $500, -0.5 for > $1,500
- **Shock events:**
  - 2021 Q1: COVID aftermath × 0.85
  - 2022 Q1-Q2: Supply chain + Ukraine war × 0.75
  - 2023 Q1: Tech layoffs × 0.90
  - 2024 Q4 iPhones: iPhone 16 boost × 1.15
- `quantity_realistic`: Negative Binomial(k=2.5, p=k/(k+mu)) + 2% zero inflation

**Phase 4 — Final Revenue:**
- `sales_amount_realistic` = quantity_realistic × price_realistic

#### `scripts/analysis_report.py`
**10-chart analysis suite (saved to `data/processed/analysis_plots/`):**
1. Monthly transaction volume (2021-2025)
2. Year-over-year monthly comparison
3. Category distribution by year (stacked bar %)
4. Top 15 products by revenue (horizontal bar)
5. Sales amount & quantity distributions (histograms)
6. Seasonality heatmap (month × year)
7. Price distribution by category (box plots)
8. Daily revenue with anomaly detection (2.5-sigma rolling)
9. Product recency validation (60/20/10/5/5 target distribution)
10. Summary statistics dashboard

#### `scripts/compare_v2_v3.py`
**8-chart before/after comparison (v2 vs v3):**
1. Seasonality — monthly avg revenue
2. Season factor distribution
3. Yearly revenue trend
4. Promo distribution by month
5. Category demand (avg quantity)
6. Daily revenue time series
7. Seasonality heatmap
8. Shock events visibility

#### `scripts/deep_diagnostic.py`
- Revenue decomposition between v2 and v3
- Factor-by-factor comparison (price, quantity, mu_demand, all 5 factors)
- Yearly breakdown and shock event impact analysis

#### `scripts/update_prices.py`
- Maps real Apple product base prices to the dataset
- Applies category-specific depreciation logic:
  - iPhones: -$100/year, floor at 40% of base
  - iPads/Watches: -15%/year
  - Macs: -10%/year
  - Audio/TV/Accessories: -5%/year
  - Services: stable pricing

#### `scripts/generate_catboost_notebook.py`
- Programmatically generates `notebooks/CATBOOST_PROPHET.ipynb`
- 630-line script that writes a 12-section fully-annotated notebook as JSON

#### `scripts/generate_combined_notebook.py`
- Generates combined CatBoost + AdaBoost notebook

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/append_products.py` | Appends new product rows to the product catalog |
| `scripts/data_cleaning.py` | Basic data cleaning utilities |
| `scripts/data_loader.py` | CSV loading helper with encoding fallback |
| `scripts/inspect_data.py` | Quick data shape/type inspection |
| `scripts/check_nb.py` | Validates notebook JSON structure |
| `scripts/verify_fix.py` | Verifies the fix script ran correctly |
| `scripts/verify_nb.py` | Verifies notebook cell outputs |
| `scripts/test_fix.py` | Unit tests for simulation fixes |
| `scripts/generate_full.py` | Triggers full pipeline run |
| `scripts/fix_chart.py` | Fixes chart formatting issues |
| `scripts/fix_missing_feature.py` | Patches missing feature columns |
| `scripts/update_rag_prompt.py` | Updates the Ollama RAG system prompt |
| `scripts/forecasting.py` | Forecasting utility (stub) |
| `scripts/visualization.py` | Visualization utility (stub) |

---

## 🧠 APP (Streamlit Viewer)

### `Streamlit Viewer/app.py` & `Streamlit Viewer/dashboard.py`
- Interactive multi-page **Streamlit Dashboard** visualizing key metrics, retail sales forecasts, store performance, market expansion priorities, and hosting the SQLite-based local SQL RAG Agent.

---

## 📄 DOCUMENTATION

### `Documentation/001_project_overview.md`
- Full project explanation and walkthrough (this document).

### `Documentation/002_graduation_project_report.md`
- Graduation project final report document.

### `Documentation/003_data_preprocessing_pipeline.md`
- Detailed documentation of the data cleaning, merging, and macro-economics preprocessing steps.

### `Documentation/004_demand_realism_strategy.md`
- 9-section strategy document outlining the demand simulation design, including a Mermaid flow diagram of the 4-phase fix pipeline and realism grading scores.

### `Documentation/005_preprocessing_analysis.md`
- Phase 2 data cleaning, GDP anomaly verification, and intermediate preprocessing analysis.

### `Documentation/006_simulation_revenue_analysis.md`
- Highlights business validation findings, such as the composition shift (AirPods vs. Mac Pros) causing v3 revenue adjustment while reflecting correct Apple proportions.

### `Documentation/007_simulation_implementation_plan.md`
- Core redesign plan identifying key simulation factors and structural issues corrected in the v2 -> v3 transition.

### `Documentation/008_graduation_project_template.md`
- Report formatting reference template.

### `Documentation/009_product_redistribution_walkthrough.md`
- Product assignment strategies and validation metrics for realistic pricing, lifecycle age, and market share distribution.

### `Documentation/010_notebook_renaming_map.md`
- Comprehensive map of original vs. numbered notebooks and documentation files.

---

## 🗃️ PRODUCT CATALOG (`data/processed/generated_products.csv`)

98 real Apple products across 10 categories:

| Category ID | Category | Examples |
|-------------|----------|---------|
| CAT-1 | Laptops | MacBook Air M1/M2, MacBook Pro 14/16-inch |
| CAT-2 | Audio | AirPods Pro, AirPods Max, HomePod mini |
| CAT-3 | Tablets | iPad 9th-10th Gen, iPad Pro M4, iPad Air M2 |
| CAT-4 | Smartphones | iPhone 11 through iPhone 17 Pro Max |
| CAT-5 | Wearables | Apple Watch Series 5-10, Ultra, SE |
| CAT-6 | Streaming Devices | Apple TV HD, Apple TV 4K |
| CAT-7 | Desktops | iMac, Mac Mini, Mac Studio, Mac Pro |
| CAT-8 | Subscriptions | Apple Music, TV+, Arcade, Fitness+, iCloud+ |
| CAT-9 | Smart Speakers | HomePod mini |
| CAT-10 | Accessories | AirTag, Magic Keyboard, Apple Pencil |

Products span launch years **2019–2025**, enabling the recency-weighted assignment strategy.

---

## 🔁 FULL DATA PIPELINE FLOW

```
raw/sales.csv (1.04M rows)
    + raw/products.csv, stores.csv, category.csv
    ↓ [001_data_loading_and_merging.ipynb]
cleaned_apple_sales.csv (140 MB)
    ↓ [004_merge_macro_and_sales.ipynb]
    + GDP, Inflation, Exchange Rate, Internet Usage
cleaned_apple_sales_enriched.csv (208 MB)
    ↓ [005_simulate_demand_factors.ipynb — Cell 31]
    + season_factor, economic_factor, promo, trend, store, mu_demand,
      quantity_realistic, price_realistic, sales_amount_realistic
cleaned_apple_sales_enriched_realistic.csv (356 MB)
    ↓ [scripts/redistribute_products.py]
    + Real Apple products (P-1 to P-98), date shift +1yr, extend to Dec 2025
cleaned_apple_sales_v2.csv (366 MB, ~1,068,918 rows)
    ↓ [scripts/fix_simulation_factors.py]
    + All 12 simulation factors rebuilt with realistic Apple-calendar logic
cleaned_apple_sales_v3.csv (379 MB)
    ↓
    ├── [notebooks/011_catboost_forecasting.ipynb] → Time-series forecasting
    ├── [scripts/analysis_report.py] → 10 analysis charts
    ├── [scripts/compare_v2_v3.py] → Before/after validation
    └── [Streamlit Viewer/app.py] → Streamlit Dashboard
```

---

## 📊 KEY METRICS (v3 Dataset)

| Metric | Value |
|--------|-------|
| Total rows | ~1,068,918 |
| Date range | 2021-01-01 → 2025-12-31 |
| Unique products | 98 |
| Unique stores | 75 |
| Unique countries | 19 |
| Total realistic revenue | ~$5.22B |
| Avg transaction value | ~$4,882 |
| Promo rate | ~15.8% |
| Season factor unique values | 12 (one per month) |
| Zero-quantity transactions | ~2% |

---

## 🔧 TECH STACK

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data | Pandas, NumPy |
| ML Models | CatBoost, AdaBoost (scikit-learn), RandomForest |
| Clustering | K-Means (scikit-learn) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| RAG Agent | Ollama (local LLM) + SQLite |
| Notebooks | Jupyter |
| Version Control | Git, GitHub |

---

## 🚀 HOW TO RUN

```bash
# 1. Clone
git clone https://github.com/AliSherif2303/Apple-Retail-Sales-Forcasting.git
cd Apple-Retail-Sales-Forcasting

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run simulation fix (v2 → v3)
python scripts/fix_simulation_factors.py

# 5. Generate analysis charts
python scripts/analysis_report.py

# 6. Compare v2 vs v3
python scripts/compare_v2_v3.py

# 7. Run market expansion & validation
python notebooks/017_target_market_validator.py

# 8. Run Streamlit app
cd "Streamlit Viewer"
streamlit run app.py
```

---

## 📈 BUSINESS INSIGHTS

### Revenue Seasonality
- **Peak months:** December (1.45×), November (1.30×), September (1.25× — iPhone launch)
- **Trough months:** February (0.80×), July (0.87×), August (0.90×)

### Historical Shocks Modeled
- 2021 Q1: COVID aftermath (−15%)
- 2022 Q1-Q2: Supply chain + Ukraine war (−25%)
- 2023 Q1: Tech sector pullback (−10%)
- 2024 Q4 iPhones: iPhone 16 boost (+15%)

### Market Expansion Priorities
- **Immediate entry:** High-GDP, large-population, high-internet countries
- **Top city targets for new stores:** Tokyo, Seoul, Shanghai, Beijing
- **Avoid:** Saturated micro-markets (Bondi, Cupertino)

### Yearly Revenue Trajectory (v3)
| Year | Revenue |
|------|---------|
| 2021 | $0.72B |
| 2022 | $0.80B |
| 2023 | $1.04B |
| 2024 | $1.19B |
| 2025 | $1.47B |

Revenue grows from $0.72B to $1.47B (2021→2025), reflecting real-world Apple growth trends.
