# Useless Files Audit & Repository Clean-up Analysis

I have completed a thorough audit of the repository to identify files that are obsolete, redundant, broken, or temporary build outputs. Cleaning these up will drastically reduce the repository size (especially removing large unreferenced CSVs and binary Power BI backups from Git history) and make the codebase clean and professional.

---

## 🚨 1. High-Priority Clean-up Candidates (Git Bloat)
These files/folders are currently **tracked and committed to Git**, which bloats the repository size and slow down cloning/pulling. Storing large datasets or binary files directly in Git is discouraged.

| File / Folder Path | Size | Reason / Status | Suggested Action |
| :--- | :--- | :--- | :--- |
| `fact_sales.csv` (Root) | **75.1 MB** | **Unreferenced.** A duplicate of transactional sales data. The Streamlit app reads from `cleaned_apple_sales_enriched_realistic/Fact_Sales.csv` (223 MB, which has the correct product IDs), and the models run on `cleaned_apple_sales_v3.csv`. | **Delete from Git** |
| `old gr.pbix` (Root) | **63.6 MB** | **Unreferenced.** An old backup of the Power BI dashboard. Storing large binary files in Git history is extremely inefficient. | **Delete from Git** |
| `test1.pbix` (Root) | **37.3 MB** | **Test File.** Likely a test dashboard version. The Power BI dashboard design is documented in `powerbi_dashboard_design.md`. | **Delete from Git** (or move to local untracked storage if still needed) |
| `catboost_info/` (Root & App folder) | — | **Auto-generated training logs** (`time_left.tsv`, `catboost_training.json`, etc.). They are currently committed to Git because they are not ignored. | **Delete & add to `.gitignore`** |
| `notebooks/lightning_logs/` | — | **Auto-generated training logs** created by PyTorch Lightning. They should not be committed to Git. | **Delete & add to `.gitignore`** |

---

## 🛠️ 2. Temporary Dumps & Obsolete Scripts
These are scratch text files, debugging scripts, or helper scripts that are either empty stubs or point to old/deleted notebooks, meaning they will crash if executed.

### 📝 Root-Level Temporary Dumps
* `_nb_structure.txt` (3.4 KB) — Temporary cell structure dump from combined notebook generation. **(Delete)**
* `_verify.txt` (10 KB) — Copy of functions (`evaluate_model`, `plot_forecast`) dumped for validation. **(Delete)**
* `_verify2.txt` (10 KB) — Another copy of functions. **(Delete)**
* `_verify3.txt` (10.1 KB) — Another copy of functions. **(Delete)**
* `nb_summary.txt` (16.6 KB) — Code cell dumps from a preprocessing notebook. **(Delete)**

### 🐍 Obsolete Python Scripts
* `fix_chart.py` (Root, 3.7 KB) — Tries to patch `notebooks/COMBINED_ADABOOST_CATBOOST.ipynb` (which does not exist). **(Delete)**
* `fix_missing_feature.py` (Root, 1.5 KB) — Tries to patch `notebooks/COMBINED_ADABOOST_CATBOOST.ipynb` (which does not exist). **(Delete)**
* `scripts/check_nb.py` (733 B) — Checks cells of the non-existent `COMBINED_ADABOOST_CATBOOST.ipynb` and writes to `scripts/nb_output.txt`. **(Delete)**
* `scripts/verify_fix.py` (757 B) — Tries to verify fixes on the non-existent `COMBINED_ADABOOST_CATBOOST.ipynb`. **(Delete)**
* `scripts/verify_nb.py` (616 B) — Tries to verify outputs on `notebooks/CATBOOST_PROPHET.ipynb` (which does not exist). **(Delete)**
* `scripts/visualization.py` (0 B) — Empty stub file. **(Delete)**
* `scripts/forecasting.py` (0 B) — Empty stub file. **(Delete)**

### 🗒️ Scripts Folder Temporary Files
* `scripts/nb_002_source.txt` (31.7 KB) — Code cell source dump. **(Delete)**
* `scripts/nb_check_output.txt` (1.9 KB) — Temporary cell checking output (UTF-16LE). **(Delete)**
* `scripts/nb_output.txt` (1.8 KB) — Output dump from `check_nb.py`. **(Delete)**
* `scripts/nb_preview.txt` (3.7 KB) — Temporary preview dump. **(Delete)**
* `scripts/data_info.txt` (67 B) — Just contains the command `streamlit run cleaned_apple_sales_enriched_realistic/dashboard.py`. **(Delete)**
* `scripts/data_info2.txt` (1.4 KB) — Contains hardcoded dataset summary stats. **(Delete)**

---

## 👥 3. Redundant Databases & Duplicates

| File / Folder Path | Size | Reason / Status | Suggested Action |
| :--- | :--- | :--- | :--- |
| `apple_sales_rag_ollama.db` (Root) | **0 bytes** | **Empty.** The actual SQLite database files are `notebooks/apple_sales_rag_ollama.db` (99.1 MB) and `cleaned_apple_sales_enriched_realistic/rag_agent.db` (99.1 MB). | **Delete** |
| `dim_economic.csv` (Root) | 2.0 MB | **Unreferenced.** Duplicate of macroeconomic data. | **Delete** |
| `dim_product.csv` (Root) | 9 KB | **Unreferenced.** Duplicate of product catalog. | **Delete** |
| `dim_store.csv` (Root) | 3.4 KB | **Unreferenced.** Duplicate of store catalog. | **Delete** |
| `Documentation/Graduation Template - Copy.docx` | 11.7 KB | **Duplicate.** A backup copy of the Graduation Template Word document. | **Delete** |

---

## 💾 4. Intermediate Local Data Files (Git-Ignored)
These files are ignored by `.gitignore` (they won't get pushed to GitHub) but they are occupying **~500 MB of local disk space** as old intermediate states:

* `data/processed/cleaned_apple_sales_BEFORE_ENRICH_BACKUP.csv` (**139.7 MB**) — Backup. **(Delete)**
* `data/processed/cleaned_apple_sales_changes.zip` (**159.3 MB**) — Zip backup. **(Delete)**
* `data/processed/cleaned_apple_sales_enriched.csv` (**207.6 MB**) — Intermediate pre-simulation data. **(Delete)**

---

## 🏷️ 5. Code Style & Naming Adjustments
Some active files have irregular or temporary-sounding names that deviate from the documentation or standard Python practices:

1. **`scripts/v3 & v2 datasets fixing.py`**
   * **Status:** **Active.** This is the core script that fixes the simulation factors (v2 → v3).
   * **Issue:** The filename contains spaces and an ampersand (`&`), making execution via CLI clunky. The docstring inside the script references the original name: `fix_simulation_factors.py`.
   * **Recommendation:** Rename it back to **`scripts/fix_simulation_factors.py`** and update references in `README.md` and `full_explanation.md` (which already call it `fix_simulation_factors.py`).
2. **`notebooks/new new target.py`**
   * **Status:** **Active.** This contains the professional country-level market validation logic (weighted scores, sales estimates, and risk analysis).
   * **Issue:** The name `new new target.py` is extremely informal.
   * **Recommendation:** Rename it to **`notebooks/target_market_validator.py`** to match its purpose and look professional.
