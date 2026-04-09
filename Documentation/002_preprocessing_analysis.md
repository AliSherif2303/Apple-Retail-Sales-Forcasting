# Documentation: 002_preprocess_externals.ipynb

This document outlines the end-to-end data processing, simulation, and feature engineering pipeline executed in the `002_preprocess_externals.ipynb` notebook. The primary goal of this notebook is to clean external macroeconomic factors, inject realistic retail patterns into the sales data, and prepare time-series features for future forecasting models.

## 1. Initial Data Cleaning & Anomaly Correction
The notebook begins by loading the `cleaned_apple_sales_enriched.csv` dataset and inspecting the external metrics appended to it.

**Key Insight (GDP Anomaly):** 
The analysis revealed that certain countries (specifically **South Korea** and **Taiwan**) had suspiciously large `gdp` values exceeding 200,000. It was uncovered that these values represented the **Total GDP** (in the trillions/billions) rather than the **GDP per capita** used for the rest of the dataset.

**Fix Applied:**
* A manual lookup table was injected for Korea and Taiwan (ranging from 2020-2024) containing their actual GDP per capita (~$28,000 - $35,000).
* The anomalies were overwritten using this lookup, and the column was officially renamed to `gdp_per_capita`.
* Null values in the `internet_usage_pct` column were filled using the grouped mean of the specific country and year.

## 2. Simulating Retail "Realism" (Data Augmentation)
Since the base dataset lacked natural retail volatility, an extensive mathematical simulation was applied to generate realistic variables: `quantity_realistic`, `price_realistic`, and `sales_amount_realistic`.

**Mathematical Factors Injected:**
1. **Seasonal Factor:** Simulated high demand for holidays (November = 1.3x, December = 1.5x) and slow periods (January = 0.8x).
2. **Economic Factor:** Demand was dynamically scaled based on GDP per capita (purchasing power) and negatively impacted by the `inflation_rate`.
3. **Promotion Spikes:** Randomly introduced a 10% chance of a "promotion day", which boosts demand by 1.4x and reduces prices by 15%.
4. **Dynamic Pricing:** Prices were adjusted to float with inflation rates and a 3% random normal volatility.
5. **Product Lifecycle Trend:** Calculates the `days_from_start` for each product and assigns a slight positive or negative long-term trend multiplier.
6. **Store Heterogeneity:** Assigned unique normal distributions to different `store_id`s to represent high-traffic vs. low-traffic locations.
7. **Price Elasticity:** Simulated a negative relationship between price hikes and demand drops, operating at a high elasticity factor (-1.1).
8. **Marco-Economic Shock:** Manually injected a 30% drop in demand (`mu_demand` *= 0.7) for the window between **March 1, 2022, and June 1, 2022**.

**Final Target Generation:**
* The final quantity was drawn from a **Negative Binomial Distribution** (dispersion parameter $k=2.8$) based on the combined `mu_demand` factor, effectively recreating realistic, discrete, and over-dispersed customer counts.
* A 2% chance of absolute zero sales (Zero-Inflation) was added.
* Extreme outlier quantities were safely capped at the 99.5th percentile.
* The output dataset was exported to `cleaned_apple_sales_enriched_realistic.csv`.

## 3. Exploratory Data Analysis (EDA) Insights
Throughout the file, histograms, heatmaps, and line charts verified the simulations:
* The original vs. realisitic quantity distribution showed that the new distribution captures a natural "long-tail" typical of retail items, dropping off smoothly.
* A strong negative correlation trend between `price_realistic` and `quantity_realistic` visibly validated the injected price elasticity module.
* Store sales appropriately clustered into varying performance cohorts rather than flat uniform averages.

## 4. Time-Series Aggregation & Feature Engineering
With the realistic data generated, the notebook concludes by structuring it for Machine Learning models (like XGBoost or LSTMs).

**Daily Matrix:**
* Grouped by `sale_date`, `store_id`, `product_id`.
* Any missing daily pricing gaps were patched utilizing forward-fill (`ffill`) and backward-fill (`bfill`).
* Saved as `cleaned_apple_sales_daily.csv`.

**Monthly Forecasting Matrix:**
The data was aggregated to a Monthly Frequency ('ME') per store and country to build the final prediction arrays. The following advanced features were built inline:
* **Lagged Features (`lag_1, lag_2, lag_3, lag_6, lag_12`):** What were the sales 1, 3, or 12 months ago?
* **Rolling Statistics:** 3-month rolling mean and 6-month rolling standard deviation (trend volatility).
* **Cyclical Time:** Deconstructed the `month` integer into `month_sin` and `month_cos` so algorithms understand December (12) is right next to January (1).
* **Logarithmic Target:** Created a `target_log` using `np.log1p()` on the sales amount to stabilize variance and help regression models converge better.
