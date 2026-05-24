# Pipeline & Documentation Renaming Map: Before & After

To organize the project logically, all notebooks in the `notebooks/` directory and documentation files in the `Documentation/` directory have been renamed using sequential numbered prefixes.

---

## 📓 1. Notebooks Renaming Table

| Original Filename | New Numbered Filename | Logical Phase / Purpose |
| :--- | :--- | :--- |
| `01_data_loading_and_merging.ipynb` | `001_data_loading_and_merging.ipynb` | Ingestion and initial merging of core CSVs |
| `02_exploratory_data_analysis.ipynb` | `002_exploratory_data_analysis.ipynb` | Initial EDA on raw transactional data |
| `001_preprocess_externals.ipynb` | `003_preprocess_external_macro_data.ipynb` | Cleaning and normalizing external GDP/Inflation data |
| `002_merge_enrich.ipynb` | `004_merge_macro_and_sales.ipynb` | Merging sales transactions with external macro indicators |
| `002_preprocess_externals.ipynb` | `005_simulate_demand_factors.ipynb` | Running the core 12-factor demand simulation |
| `003_preprocessing_final.ipynb` | `006_final_preprocessing_features.ipynb` | Final preprocessing and cleaning before ML modeling |
| `preprocessing_merged_enriched.ipynb` | `007_preprocessing_merged_enriched_archive.ipynb` | Old/large merged preprocessing scratchpad (archived) |
| `shifting.ipynb` | `008_date_shifting_utility.ipynb` | Utility notebook for shifting date ranges |
| `ADABOOST.ipynb` | `009_adaboost_forecasting.ipynb` | Training and tuning the AdaBoost Regressor |
| `ADA_LIVE.ipynb` | `010_adaboost_with_baselines.ipynb` | Tuned AdaBoost model compared with Decision Tree baseline |
| `CATBOOST.ipynb` | `011_catboost_forecasting.ipynb` | Training and tuning the CatBoost Regressor |
| `CAT_LIVE.ipynb` | `012_catboost_with_baselines.ipynb` | Tuned CatBoost model compared with Linear Regression baseline |
| `cat&ada.ipynb` | `013_combined_ada_and_catboost.ipynb` | Side-by-side comparison of AdaBoost and CatBoost |
| `EDA_NEW_DATASET.ipynb` | `014_exploratory_data_analysis_enriched.ipynb` | EDA on the final enriched/realistic v3 dataset |
| `Ollama_SQL_RAG_Agent.ipynb` | `015_ollama_sql_rag_agent.ipynb` | SQLite database schema and Ollama local LLM RAG agent |
| `market_expansion_stores.ipynb` | `016_market_expansion_store_clustering.ipynb` | K-Means clustering and store expansion priority analysis |
| `new new target.py` | `017_target_market_validator.py` | Python script for country Opportunity scoring & sales estimation |

---

## 📄 2. Documentation Renaming Table

| Original Filename | New Numbered Filename | Logical Phase / Purpose |
| :--- | :--- | :--- |
| `full_explanation.md` | `001_project_overview.md` | Full project explanation and walkthrough |
| `Graduation_Project_Report.md` | `002_graduation_project_report.md` | Graduation project final report |
| `Data_Preprocessing.md` | `003_data_preprocessing_pipeline.md` | Detailed preprocessing phase documentation |
| `realism_strategy.md` | `004_demand_realism_strategy.md` | Realism/simulation strategy overview |
| `002_preprocessing_analysis.md` | `005_preprocessing_analysis.md` | Phase 2 data cleaning & GDP anomaly report |
| `revenue_analysis.md` | `006_simulation_revenue_analysis.md` | Explains v3 composition shift & product sales differences |
| `implementation_plan.md` | `007_simulation_implementation_plan.md` | Designed fixes proposal plan for simulation factors |
| `Graduation Template.md` | `008_graduation_project_template.md` | Report formatting template |
| `walkthrough.md` | `009_product_redistribution_walkthrough.md` | Product redistribution strategy & validation charts |
| `notebook_renaming_map.md` | `010_notebook_renaming_map.md` | Comprehensive map of original vs. numbered notebooks and documentation files |
| `useless_files_analysis.md` | `011_useless_files_analysis.md` | Report detailing the audit and cleanup of obsolete files |
| `[New File]` | `012_antigravity_actions_2026_05_24.md` | Summary report of AI actions, file renames, risks, and guidelines |

---

## ⚠️ Important Note on Non-Notebook Assets
* The trained model binary files (`ADA_LIVE.joblib` and `CAT_LIVE.cbm`) and the RAG database (`apple_sales_rag_ollama.db`) **remain unchanged** in the `notebooks/` folder.
* *Reason:* Streamlit dashboard files (such as `2_Long_Term_Forecasting.py` and `3_Short_Term_Forecasting.py`) load these specific filenames, and renaming them would break the active dashboard application.
