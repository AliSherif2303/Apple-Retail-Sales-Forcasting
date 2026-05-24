# 🍏 Apple Retail Sales Forecasting — End-to-End ML Pipeline & Dashboard

An end-to-end **retail sales intelligence system** for Apple's global store network (75 stores across 19 countries). This project cleans raw transaction data, enriches it with macroeconomic indicators, simulates realistic demand behaviors, trains time-series forecasters (CatBoost and AdaBoost), performs clustering-based market expansion validation, and deploys an interactive Streamlit viewer containing a local SQL RAG Agent.

---

## 👨‍💻 Project Team (Graduation Project)
* **Ali Sherif Salaheldin** — Project Owner & Lead Machine Learning Engineer
* **Ali Mohamed** — ML Engineer
* **Hassan saad** — MLOps Engineer
* **Ahmed adel** — Data Analyst
* **Mohammed azzam** — Data Engineer

---

## 📁 Repository Structure

```
Apple-Retail-Sales-Forcasting/
│
├── data/
│   ├── raw/                         ← Original Kaggle sales, products, stores, warranty CSVs
│   └── processed/                   ← Normalised macro data & cleaned datasets (v1 to v3)
│
├── notebooks/                       ← Numbered ML & analytics pipeline files
│   ├── 001_data_loading_and_merging.ipynb
│   ├── 002_exploratory_data_analysis.ipynb
│   ├── 003_preprocess_external_macro_data.ipynb
│   ├── 004_merge_macro_and_sales.ipynb
│   ├── 005_simulate_demand_factors.ipynb
│   ├── 006_final_preprocessing_features.ipynb
│   ├── 007_preprocessing_merged_enriched_archive.ipynb
│   ├── 008_date_shifting_utility.ipynb
│   ├── 009_adaboost_forecasting.ipynb
│   ├── 010_adaboost_with_baselines.ipynb
│   ├── 011_catboost_forecasting.ipynb
│   ├── 012_catboost_with_baselines.ipynb
│   ├── 013_combined_ada_and_catboost.ipynb
│   ├── 014_exploratory_data_analysis_enriched.ipynb
│   ├── 015_ollama_sql_rag_agent.ipynb
│   ├── 016_market_expansion_store_clustering.ipynb
│   └── 017_target_market_validator.py
│
├── scripts/                         ← Pipeline automation & feature-engineering scripts
│   ├── fix_simulation_factors.py    ← Rebuilds the 12-factor demand simulation (v2 → v3)
│   ├── redistribute_products.py     ← Maps real product IDs and shifts timeline (v1 → v2)
│   ├── analysis_report.py           ← Generates 10 diagnostics validation plots
│   ├── compare_v2_v3.py             ← Compares simulation strategy enhancements
│   └── update_prices.py             ← Applies product depreciation calculations
│
├── Streamlit Viewer/                ← Multi-page Streamlit Dashboard app
│   ├── app.py                       ← Main entrypoint
│   ├── dashboard.py                 ← Landing page & general KPIs
│   ├── requirements.txt             ← App dependencies (including catboost)
│   ├── Dockerfile & docker-compose.yml
│   └── pages/                       ← Core modules (Forecasting, Clustering, RAG)
│
├── Documentation/                   ← Numbered project reports and action briefs
│   ├── 001_project_overview.md
│   ├── 002_graduation_project_report.md
│   ├── 003_data_preprocessing_pipeline.md
│   ├── 004_demand_realism_strategy.md
│   ├── 005_preprocessing_analysis.md
│   ├── 006_simulation_revenue_analysis.md
│   ├── 007_simulation_implementation_plan.md
│   ├── 008_graduation_project_template.md
│   ├── 009_product_redistribution_walkthrough.md
│   ├── 010_notebook_renaming_map.md
│   ├── 011_useless_files_analysis.md
│   └── 012_antigravity_actions_2026_05_24.md
│
├── requirements.txt                 ← Core pipeline python dependencies
└── README.md                        ← Main documentation (this file)
```

---

## 🔁 Data & Pipeline Flow

The system processes data from raw ingestion to downstream forecasts through the following stages:

```
[Kaggle Sales CSV] + [Product Catalog] + [Store Locations]
                     ↓
         [001_data_loading_and_merging.ipynb]
                     ↓
          cleaned_apple_sales.csv (v1)
                     ↓
       [003_preprocess_external_macro_data.ipynb]
     + Adds GDP, Inflation, Exchange, and Internet Usage
                     ↓
          cleaned_apple_sales_enriched.csv
                     ↓
       [005_simulate_demand_factors.ipynb]
     + 12-Factor Demand Simulation Equations
                     ↓
   cleaned_apple_sales_enriched_realistic.csv
                     ↓
         [scripts/redistribute_products.py]
     + Maps real Apple product IDs & shifts timeline to 2021-2025
                     ↓
          cleaned_apple_sales_v2.csv
                     ↓
       [scripts/fix_simulation_factors.py]
     + Corrects season factors, country-level GDP caps, and shock events
                     ↓
          cleaned_apple_sales_v3.csv (Final Dataset)
                     ↓
         ┌───────────┴───────────┐
         ▼                       ▼
 [Machine Learning Models]   [Streamlit Viewer App & SQL RAG Agent]
 CatBoost / AdaBoost        Interactive Forecasting, Clustering & LLM
```

---

## 🛠️ Setup & Running Instructions

Follow these instructions to run the pipeline, start the Streamlit viewer, and spin up the SQL RAG Agent locally.

### 1️⃣ Clone & Environment Setup
Clone the repository and install the dependencies into a python virtual environment:
```powershell
# Clone the repository
git clone https://github.com/AliSherif2303/Apple-Retail-Sales-Forcasting.git
cd Apple-Retail-Sales-Forcasting

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # On Windows (PowerShell)
source .venv/bin/activate     # On macOS/Linux

# Install core pipeline dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Data Pipeline (v2 → v3 Simulation)
Generate the final production dataset by applying corrected seasonality, economic shocks, and price depreciation formulas:
```powershell
python scripts/fix_simulation_factors.py
```
*(Optional: Run `python scripts/analysis_report.py` to output 10 diagnostic charts under `data/processed/analysis_plots/`).*

### 3️⃣ Run local Ollama (For SQL RAG Agent)
The SQL RAG agent queries your sales database using a local, secure LLM.
1. Download and install Ollama from **[ollama.com](https://ollama.com)**.
2. Open a new PowerShell window and download the optimized coder model:
   ```powershell
   ollama pull qwen2.5-coder:3b
   ```

### 4️⃣ Run Streamlit Dashboard
Navigate to the `Streamlit Viewer` folder, install its specific requirements, and start the application:
```powershell
cd "Streamlit Viewer"
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧠 Dashboard Modules (`Streamlit Viewer/`)

Once launched, the dashboard exposes 7 interactive modules:
1. **🏠 Executive Dashboard (`dashboard.py`)**: General metrics, sales trends, geographic map distributions, and product category breakdowns.
2. **📈 Long-Term Forecasting**: Horizon-based monthly projections using trained regressors.
3. **📉 Short-Term Forecasting**: Real-time sales predictions for upcoming weeks.
4. **🎯 Market Entry Clustering**: Visualizes global city classifications (Untapped Megacities, Wealthy Core, Saturated Niches, Emerging Mid-Tier) using K-Means.
5. **🏙️ Target Market Validator**: Opportunity score evaluator ranking countries based on purchasing power, digital reach, and population sizes.
6. **🤖 SQL RAG Agent**: Translates natural English questions (e.g., *"How many units did we sell in Japan in Q4 2024?"*) into secure DuckDB SQL queries using `qwen2.5-coder:3b` and outputs executive briefings.
7. **🔮 Live Model Forecast**: Interactive sandbox running CatBoost vs. AdaBoost side-by-side on any store and horizon.

---

## ⚠️ local-only & Ignored Files Alert
To save storage space on GitHub and keep your pulls/pushes running at high speeds, large and binary files have been added to `.gitignore` and **removed from remote tracking, but are kept locally in your folder**:
* **Power BI Dashboards**: `old gr.pbix` (63.6 MB) and `test1.pbix` (37.3 MB).
* **Large Datasets**: `fact_sales.csv` (75.1 MB) in the project root.
* **Model Logs**: `catboost_info/` and PyTorch Lightning logs `notebooks/lightning_logs/`.
* *Caution*: Never run `git clean -fdx` or force-clean commands, as it will delete these local files from your workspace.
