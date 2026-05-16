# Apple Retail Sales Analysis & Forecasting System

## Graduation Project — Data Science Major

---

## 1. Title Page

- **Project Title:** Apple Retail Sales Analysis & Forecasting System
- **Institution:** Faculty of Computer Science & Artificial Intelligence
- **Program:** Data Science Major
- **Date:** May 2026

### Team Members
| Name |
|------|
| Ali Sherif |
| Ali Mohamed |
| Ahmed Adel |
| Hassan Saad |
| Mohamed Azzam |

### Supervisors
- **Dr. Christine Albert**
- **Dr. Antony**

---

## 2. Abstract

This project presents an end-to-end retail sales analysis and forecasting system for Apple's global store network spanning 75 stores across 19 countries. The system addresses the challenge of predicting future sales revenue at the store level by ingesting raw transactional data (~1.04 million records), enriching it with macroeconomic indicators (GDP, inflation, exchange rates, internet penetration), and simulating realistic retail demand patterns including seasonality, promotions, price elasticity, and historical shock events. Two ensemble machine learning models — CatBoost and AdaBoost — were trained on 52 engineered time-series features to forecast monthly store-level sales. CatBoost achieved an R² of 0.9784 with MAE of $39,896, while AdaBoost achieved R² of 0.9032 with MAE of $94,796. The project also includes K-Means market clustering for store expansion planning, a country-level market entry scoring system, an AI-powered SQL RAG agent using Ollama for natural language data querying, and a multi-page Streamlit dashboard for interactive visualization. The final enriched dataset (v3) contains ~1,068,918 transactions covering 2021–2025 with $5.22 billion in total realistic revenue across 98 real Apple products and 10 categories. A Power BI dashboard was also designed for executive-level reporting.

---

## 3. Introduction

### Problem Statement
Retail sales forecasting is critical for inventory management, staffing, and strategic planning. Apple operates hundreds of retail stores worldwide, each with unique demand patterns driven by local economic conditions, seasonal trends, product launch cycles, and promotional activities. Accurately forecasting store-level sales enables better resource allocation, reduces overstock/understock risks, and supports data-driven expansion decisions. This project addresses the need for a comprehensive, data-driven forecasting system that accounts for these complex, multi-dimensional factors.

### Objectives
1. Build a complete data pipeline from raw Kaggle sales data to a production-ready enriched dataset with realistic retail patterns.
2. Train and evaluate time-series forecasting models (CatBoost, AdaBoost) for monthly store-level sales prediction.
3. Perform market expansion analysis using clustering and scoring to identify optimal locations for new Apple Stores.
4. Develop an interactive Streamlit dashboard for real-time forecasting and data exploration.
5. Implement an AI-powered SQL RAG agent for natural language querying of sales data.
6. Create Power BI dashboards for executive-level business intelligence.

### Research Questions
1. Can ensemble ML models accurately forecast monthly Apple Store revenue using time-series features and macroeconomic indicators?
2. Which factors (seasonality, promotions, economic conditions, product lifecycle) most significantly influence Apple retail sales?
3. Which global cities and countries represent the highest-priority targets for Apple Store expansion?
4. How do CatBoost and AdaBoost compare in retail sales forecasting accuracy and stability?

### Significance
This project provides a reusable framework for retail sales forecasting that combines traditional time-series analysis with modern ML techniques. It demonstrates how enriching transactional data with external economic indicators improves forecast accuracy. The market expansion module offers actionable business intelligence for strategic decision-making, while the RAG agent showcases the integration of LLMs with structured data for accessible analytics.

---

## 4. Literature Review

### Time-Series Forecasting in Retail
Traditional retail forecasting relies on ARIMA, SARIMA, and exponential smoothing methods. Recent advances in gradient boosting (XGBoost, LightGBM, CatBoost) have shown superior performance on tabular time-series data by capturing non-linear relationships and feature interactions. CatBoost, in particular, handles categorical features natively and uses ordered boosting to reduce overfitting.

### Macroeconomic Factors in Sales Prediction
Research shows that GDP per capita, inflation rates, and exchange rates significantly influence consumer purchasing behavior, especially for premium products. Apple's pricing power — demonstrated by near-zero correlation between inflation and sales volume in our data — represents an interesting finding that aligns with luxury brand economics literature.

### Market Expansion Analytics
K-Means clustering and scoring models have been widely used for market segmentation and site selection. Our approach combines population data, GDP, internet penetration, and existing store performance to create a multi-criteria expansion framework.

### Gaps Addressed
- Most retail forecasting studies use a single model; we compare CatBoost vs. AdaBoost with per-store metrics across 75 stores.
- Few projects combine forecasting with market expansion analysis and interactive AI querying in a single system.
- Our demand simulation pipeline (Negative Binomial with category-aware scaling, shock events, and price elasticity) creates more realistic training data than simple random augmentation.

---

## 5. Methodology

### 5.1 Data Collection

#### Primary Dataset
Source: [Apple Retail Sales Dataset — Kaggle](https://www.kaggle.com/datasets/amangarg08/apple-retail-sales-dataset)

| File | Description | Shape |
|------|-------------|-------|
| `sales.csv` | Sales transactions (sale_id, store_id, product_id, quantity, sale_date) | ~1,040,200 rows |
| `products.csv` | Product catalog with IDs, categories, launch dates, prices | 89 rows |
| `stores.csv` | 75 Apple Stores — name, city, country | 75 rows |
| `category.csv` | 10 product categories (CAT-1 to CAT-10) | 10 rows |
| `warranty.csv` | 30,000 warranty claims & repair statuses | 30,000 rows |

#### External Macroeconomic Data
| File | Description |
|------|-------------|
| `gdp.csv` | GDP per capita by country/year |
| `infilation.csv` | Annual inflation rates by country |
| `exchange.csv` | Currency exchange rates |
| `Individuals using the Internet.csv` | Internet penetration % by country |
| `Year-Category-Product-StartingPrice.csv` | Historical Apple product pricing |

#### Ethical Considerations
- All data is publicly available from Kaggle (open license).
- No personally identifiable information (PII) is present in the dataset.
- Macroeconomic data is sourced from public World Bank indicators.
- Demand simulation was applied transparently with documented methodology.

---

### 5.2 Data Preprocessing

#### Data Cleaning
- **GDP Anomaly Fix:** South Korea and Taiwan had GDP values >200,000 (Total GDP instead of GDP per capita). Manually corrected using lookup tables with actual per-capita values (~$28K–$35K).
- **Missing Values:** `internet_usage_pct` nulls filled using grouped mean by country and year.
- **Column Cleanup:** Dropped redundant columns (`gdp`, `gdp_cleaned`, `country`, `country_norm`).
- **Date Handling:** All dates converted to `datetime64`, date range shifted +1 year (2020–2024 → 2021–2025) for temporal relevance.

#### Feature Engineering — Simulation Pipeline (v2 → v3)
A 4-phase simulation pipeline rebuilt 12 derived columns with realistic Apple retail patterns:

**Phase 1 — Independent Factors:**
- `season_factor`: 12 distinct monthly multipliers (Sep=1.25 iPhone launch, Dec=1.45 holiday, Feb=0.80 lowest).
- `economic_factor`: Per-country GDP normalization instead of global mean.
- `product_trend`: Category-aware lifecycle trends (Wearables +0.0004/day, Smart Speakers −0.0003/day).
- `store_factor`: Annual drift ±3% per store per year.

**Phase 2 — Dependent Factors:**
- `promo_flag/promo_factor`: 3-layer model (monthly base + product age boost + category boost). Overall ~15.8% promo rate.
- `price_realistic`: 4-layer pricing (inflation + age depreciation + promo discount + market noise).

**Phase 3 — Demand Generation:**
- `mu_demand`: Category-aware base demand (iPhones 1.4×, Mac Pro 0.5×) × price-inverse scaling.
- Price elasticity: −1.2 for <$500, −0.5 for >$1,500.
- Shock events: 2021 Q1 COVID (×0.85), 2022 Q1–Q2 supply chain (×0.75), 2023 Q1 tech layoffs (×0.90), 2024 Q4 iPhone 16 boost (×1.15).
- `quantity_realistic`: Negative Binomial (k=2.5) + 2% zero inflation + 99.5th percentile cap.

**Phase 4 — Final Revenue:**
- `sales_amount_realistic` = quantity_realistic × price_realistic.

#### Product Redistribution
98 real Apple products across 10 categories were assigned using recency-weighted distribution: 60% current-year, 20% Y−1, 10% Y−2, 5% older, 5% subscriptions. ~35,700 new rows generated for Nov 13–Dec 31, 2025.

#### Train-Test Split
Temporal hold-out: last 5 months of data used as test set. TimeSeriesSplit with 5 folds used for cross-validation to respect temporal ordering.

---

### 5.3 Exploratory Data Analysis (EDA)

#### Key Visualizations & Findings

**Seasonality Pattern:**
| Month | Avg Monthly Revenue | Index vs Mean |
|-------|-------------------|---------------|
| Feb | $57.2M | **65% (lowest)** |
| Sep | $108.0M | 123% |
| Nov | $119.9M | 137% |
| Dec | $134.5M | **153% (highest)** |

December revenue ($134.5M avg) is **2.35× February** ($57.2M), aligning with Apple's real launch + holiday cycle.

**Category Revenue Breakdown:**
| Category | Revenue | Share |
|----------|---------|-------|
| Smartphone | $2.81B | **53.8%** |
| Laptop | $647.4M | 12.4% |
| Desktop | $556.6M | 10.7% |
| Tablet | $549.5M | 10.5% |
| Wearable | $455.1M | 8.7% |
| Audio | $154.8M | 3.0% |

**Geographic Performance — Top 5 Markets:**
| Country | Revenue | Share |
|---------|---------|-------|
| United States | $1.06B | 20.3% |
| Australia | $487.3M | 9.3% |
| China | $487.1M | 9.3% |
| Japan | $435.0M | 8.3% |
| Canada | $364.4M | 7.0% |

**Promotion Effectiveness:**
Promotions deliver +25.5% lift in average transaction value ($5,889 vs $4,692 for non-promo).

**Macroeconomic Correlations:**
| Factor | Correlation with Revenue |
|--------|------------------------|
| GDP per Capita | +0.32 (weak positive) |
| Inflation Rate | −0.11 (negligible) |
| Internet Usage % | +0.05 (negligible) |

The weak GDP correlation suggests Apple's revenue is not purely a function of national wealth, confirming Apple's exceptional pricing power.

---

### 5.4 Model Selection

#### Algorithms Used

**1. CatBoost Regressor**
- Gradient boosting framework by Yandex optimized for categorical features.
- Uses ordered boosting to prevent target leakage.
- Chosen for its native handling of categorical store/product IDs and robust performance on tabular data.

**2. AdaBoost Regressor**
- Adaptive boosting with decision tree base estimators.
- Sequentially trains weak learners, focusing on hard-to-predict samples.
- Chosen as a comparison baseline with different boosting philosophy (sample reweighting vs. gradient descent).

**Justification:** Both are ensemble methods well-suited for time-series regression with engineered features. CatBoost was expected to outperform due to its advanced regularization and native categorical support. AdaBoost provides a simpler, more interpretable alternative.

**Baseline:** A naive model using `sales_lag_1` (last month's sales) as the prediction serves as the baseline comparison.

#### Feature Set (28 features after refinement, reduced from 54)
- **Lag features:** sales_lag_1, 2, 3, 6, 12
- **Rolling statistics:** 3-month and 6-month rolling means
- **Momentum:** month-over-month % change, lag1 vs roll6 ratio
- **Cyclical time:** month_sin, month_cos, is_holiday_season, is_launch_season
- **Economic:** gdp_per_capita, inflation_rate, exchange_rate, internet_usage_pct + their changes
- **Store metadata:** store_encoded, num_transactions, num_unique_products, num_categories
- **Calendar:** year

---

### 5.5 Evaluation Metrics

| Metric | Formula | Why Appropriate |
|--------|---------|-----------------|
| **MAE** | Mean Absolute Error | Interpretable in dollar terms — "average prediction error in $" |
| **RMSE** | Root Mean Squared Error | Penalizes large errors — important for detecting catastrophic mispredictions |
| **MAPE** | Mean Absolute Percentage Error | Scale-independent — enables cross-store comparison |
| **R²** | Coefficient of Determination | Measures overall variance explained — key metric for regression quality |

---

## 6. Implementation

### Tools & Technologies

| Layer | Tools |
|-------|-------|
| Language | Python 3.12 |
| Data Processing | Pandas, NumPy |
| ML Models | CatBoost, AdaBoost (scikit-learn), Random Forest |
| Clustering | K-Means (scikit-learn) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit (7-page app), Power BI |
| RAG Agent | Ollama (Qwen 2.5-Coder 3B) + SQLite + LangChain |
| Deployment | Docker, Streamlit Cloud |
| Version Control | Git, GitHub |
| Notebooks | Jupyter |

### Development Process

**Data Pipeline:**
```
raw/sales.csv (1.04M rows)
  + products, stores, category
  → cleaned_apple_sales.csv (140 MB)
  + GDP, Inflation, Exchange Rate, Internet Usage
  → cleaned_apple_sales_enriched.csv (208 MB)
  + 12 simulation factors
  → cleaned_apple_sales_enriched_realistic.csv (356 MB)
  + Real Apple products, date shift
  → cleaned_apple_sales_v2.csv (366 MB)
  + Fixed simulation factors (4-phase rebuild)
  → cleaned_apple_sales_v3.csv (379 MB, ~1,068,918 rows)
```

**Streamlit App Architecture (7 pages):**
1. Dashboard — Interactive EDA with KPIs, trends, market analysis
2. Long-Term Forecasting — 12-month recursive AdaBoost & CatBoost forecast
3. Short-Term Forecasting — Pre-computed model predictions by store
4. Market Clustering — K-Means city segmentation
5. Target Market Validator — Country-level market entry scoring
6. SQL RAG Agent — Natural language → SQL → answer pipeline
7. Live Forecast — Real-time predictions using saved .cbm / .joblib models

**Star Schema Data Model (for Dashboard & Power BI):**
- `Fact_Sales` (1,068,918 rows, 24 columns)
- `Dim_Product` (89 products, 6 columns)
- `Dim_Store` (75 stores, 4 columns)
- `Dim_Macroeconomics` (95 rows, 7 columns)

### Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| GDP anomaly (Korea/Taiwan had total GDP instead of per-capita) | Manual lookup table with corrected values |
| Simulation factors stale after product redistribution & date shift | Built `fix_simulation_factors.py` — 4-phase pipeline rebuilding all 12 columns |
| Revenue dropped 20% in v3 despite quantity increasing 24% | Verified as correct: category-aware demand scaling (Mac Pro 6→2 units, AirPods 6→11 units) |
| Feature leakage from `store_target_enc` | Removed shortcut features, reduced from 54 to 28 features |
| Large CSV files (379 MB) on GitHub | Git LFS for files >100 MB |
| Recursive forecasting instability | Careful lag/rolling feature propagation prevents error accumulation |

---

## 7. Results & Discussion

### Model Performance

| Metric | CatBoost | AdaBoost |
|--------|----------|----------|
| **R²** | **0.9784** | 0.9032 |
| **MAE** | **$39,896** | $94,796 |
| **MAE %** | **2.27%** | ~5.2% |
| **RMSE %** | **4.44%** | ~8.5% |

**CatBoost outperforms AdaBoost** across all metrics with R² of 97.84% vs 90.32%. CatBoost's MAE of ~$40K on average monthly store sales of ~$1.82M represents only a 2.27% error rate.

### Feature Importance (CatBoost)

| Feature | Importance |
|---------|------------|
| sales_lag_1 | 29% |
| sales_mom_pct | 27% |
| sales_lag_12 | 20% |

The top 3 features are all temporal patterns — recent sales, momentum, and same-month-last-year — confirming the model learns genuine seasonal and trend patterns rather than shortcuts.

### Revenue Growth Trajectory (v3)

| Year | Revenue | YoY Growth |
|------|---------|------------|
| 2021 | $717.4M | — (baseline) |
| 2022 | $801.3M | +11.7% |
| 2023 | $1.04B | +30.2% |
| 2024 | $1.19B | +14.2% |
| 2025 | $1.47B | +23.1% |

Revenue doubles from $717M to $1.47B (2021→2025), reflecting real-world Apple growth trends.

### Market Expansion Results

**K-Means Clustering (4 clusters):**
- Silhouette Score: 0.409 ✅
- Davies-Bouldin Index: 0.601 ✅
- Cluster 0: "Untapped Megacities" (Tokyo, Shanghai, Beijing)
- Cluster 1: "Wealthy Core Markets" (SF, Singapore, Sydney)
- Cluster 2: "Saturated Niches" (Bondi)
- Cluster 3: "Emerging Mid-Tier" (Seoul, Bangkok, London)

**Country-Level Scoring (max 8 points):**
- GDP >$50K → +3pts, Population >100M → +3pts, Internet >80% → +2pts
- High-priority targets: large GDP + large population + high internet penetration countries

### Promotion Impact Analysis

| Metric | Promo | No Promo | Lift |
|--------|-------|----------|------|
| Avg Transaction Value | $5,889 | $4,692 | **+25.5%** |
| % of Transactions | 15.8% | 84.2% | — |

### Limitations
1. **Simulated data:** While the enrichment pipeline creates realistic patterns, quantities and prices are modeled, not actual Apple sales figures.
2. **Static macroeconomic data:** Economic indicators are at yearly granularity — sub-annual macro analysis is limited.
3. **75 stores only:** Apple operates 500+ stores globally; our dataset covers a subset.
4. **No real-time feed:** The dashboard reads static CSV files; updates require re-running the pipeline.

---

## 8. Conclusion & Future Work

### Conclusion
This project successfully built an end-to-end Apple retail sales analysis and forecasting system. Key achievements:
- **CatBoost achieves 97.84% R²** with only 2.27% average prediction error on monthly store sales.
- **Category-aware demand simulation** produces realistic sales patterns where iPhones dominate (53.8% revenue) and expensive desktops sell fewer units.
- **Market expansion analysis** identifies Tokyo, Shanghai, and Beijing as top expansion targets using K-Means clustering and multi-criteria scoring.
- **7-page Streamlit dashboard** provides interactive forecasting, market analysis, and AI-powered natural language querying.
- The **SQL RAG agent** enables non-technical users to query sales data conversationally.

### Impact
- **Business Planning:** Store-level monthly forecasts enable optimized inventory management and staffing.
- **Strategic Expansion:** Data-driven market entry scoring reduces risk in new store location decisions.
- **Accessible Analytics:** The RAG agent democratizes data access for non-technical stakeholders.
- **Revenue Insights:** Identification of seasonal patterns (Dec is 2.35× Feb) and promotion effectiveness (+25.5% lift) directly informs marketing strategy.

### Future Work
1. Integrate real-time sales data via API for live dashboard updates.
2. Implement deep learning models (LSTM, Transformer) for comparison.
3. Add customer segmentation and product recommendation modules.
4. Expand to 500+ stores with Apple's full global network.
5. Deploy the Streamlit app on cloud infrastructure with authentication.
6. Build automated retraining pipelines for model drift detection.

---

## 9. References

1. Apple Retail Sales Dataset — Kaggle. https://www.kaggle.com/datasets/amangarg08/apple-retail-sales-dataset
2. World Bank Open Data — GDP, Inflation, Internet Usage. https://data.worldbank.org
3. CatBoost Documentation — Yandex. https://catboost.ai/docs/
4. Scikit-learn AdaBoostRegressor. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
5. Streamlit Documentation. https://docs.streamlit.io
6. LangChain SQL Agent Documentation. https://python.langchain.com
7. Ollama — Local LLM Runtime. https://ollama.com
8. Plotly Graphing Library. https://plotly.com/python/

---

## 10. Appendices

### Appendix A: GitHub Repository
https://github.com/AliSherif2303/Apple-Retail-Sales-Forcasting

### Appendix B: Product Catalog Summary
98 real Apple products across 10 categories:

| Category | Examples |
|----------|---------|
| Laptops (CAT-1) | MacBook Air M1/M2, MacBook Pro 14/16-inch |
| Audio (CAT-2) | AirPods Pro, AirPods Max, HomePod mini |
| Tablets (CAT-3) | iPad 9th–10th Gen, iPad Pro M4, iPad Air M2 |
| Smartphones (CAT-4) | iPhone 11 through iPhone 17 Pro Max |
| Wearables (CAT-5) | Apple Watch Series 5–10, Ultra, SE |
| Streaming (CAT-6) | Apple TV HD, Apple TV 4K |
| Desktops (CAT-7) | iMac, Mac Mini, Mac Studio, Mac Pro |
| Subscriptions (CAT-8) | Apple Music, TV+, Arcade, Fitness+, iCloud+ |
| Smart Speakers (CAT-9) | HomePod mini |
| Accessories (CAT-10) | AirTag, Magic Keyboard, Apple Pencil |

### Appendix C: Dataset Versions

| Version | File | Size | Description |
|---------|------|------|-------------|
| Base | cleaned_apple_sales.csv | 140 MB | Merged & cleaned |
| Enriched | cleaned_apple_sales_enriched.csv | 208 MB | + Macro indicators |
| Realistic | cleaned_apple_sales_enriched_realistic.csv | 356 MB | + Simulation factors |
| V2 | cleaned_apple_sales_v2.csv | 366 MB | + Real products & date shift |
| **V3 (Final)** | **cleaned_apple_sales_v3.csv** | **379 MB** | **+ Fixed simulation (production)** |

### Appendix D: Key Metrics Summary (V3 Dataset)

| Metric | Value |
|--------|-------|
| Total Rows | ~1,068,918 |
| Date Range | 2021-01-01 → 2025-12-31 |
| Unique Products | 98 |
| Unique Stores | 75 |
| Unique Countries | 19 |
| Total Realistic Revenue | ~$5.22B |
| Avg Transaction Value | ~$4,882 |
| Promo Rate | ~15.8% |
