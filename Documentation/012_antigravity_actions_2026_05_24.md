# 🍏 Antigravity Action Report & Safety Guidelines
**Date:** May 24, 2026  
**Author:** Antigravity (AI Coding Assistant)  
**Target Repository:** Apple Retail Sales Forecasting

---

## 📌 1. Summary of Actions Taken in This Conversation

To streamline, organize, and optimize the codebase for your graduation project, the following clean-up, renaming, and reference update operations were executed:

### 🧹 Git Bloat Mitigation & Ignored Candidates
* **Untracked Large Datasets & Backups**: Large files such as `fact_sales.csv` (75.1 MB), `old gr.pbix` (63.6 MB), and `test1.pbix` (37.3 MB) were removed from Git tracking (`git rm --cached`) to decrease repository clone/push times. They have been added to `.gitignore` and **still exist on your local disk** for your use.
* **Ignored Training Logs**: Added auto-generated training logs (`catboost_info/` and `notebooks/lightning_logs/`) to `.gitignore` to prevent cluttering the Git index.
* **Deleted Obsolete Helper Scripts & Temporary Dumps**: Staged the cleanup of redundant stubs and dump text files (`_nb_structure.txt`, `_verify.txt`, `_verify2.txt`, `_verify3.txt`, `nb_summary.txt`, `fix_chart.py`, `scripts/check_nb.py`, etc.) that were crashing or unreferenced.

### 🏷️ Systematic Renaming
* **Notebooks Renamed (Sequential Numbering)**: Renamed 16 notebooks in `notebooks/` to a structured `001_` through `016_` format, separating EDA, preprocessing, and model training.
* **Target Market Validator**: Renamed `new new target.py` to `017_target_market_validator.py` and placed it in the notebooks folder.
* **Documentation Renamed (Sequential Numbering)**: Numbered all core files in `Documentation/` (`001_project_overview.md` to `011_useless_files_analysis.md`) so they follow a reading sequence.
* **Fix Script**: Renamed `v3 & v2 datasets fixing.py` to `fix_simulation_factors.py` to remove spaces and special characters (`&`) that cause errors in command line interpreters (e.g., PowerShell).
* **Streamlit App Directory**: Staged the directory rename of the Streamlit application from `cleaned_apple_sales_enriched_realistic/` to a user-friendly name: `Streamlit Viewer/`.

### 🔗 Reference & Link Alignment
* Updated references across documentation, specifically inside `001_project_overview.md`, to correctly reference the new numbered notebooks, the new script filenames, the renamed `Streamlit Viewer/` folder, and its proper execution steps.
* Fixed the CLI copy commands in the live forecast page of the Streamlit app to reference `"Streamlit Viewer"` instead of the old folder name.

---

## 🛡️ 2. Streamlit Viewer Safety Analysis & Integrity

You mentioned you do not want the `Streamlit Viewer` folder to have issues. Here is a safety breakdown of why the dashboard will run successfully:

### 📂 Relative Imports & Folder Relocation
Because the entire folder `cleaned_apple_sales_enriched_realistic/` was renamed to `Streamlit Viewer/` as a single unit, all internal relative structures within the folder are preserved.
* **Docker Compose Bind Mounts**: The `docker-compose.yml` mounts:
  * `.:/app` (current folder `Streamlit Viewer`)
  * `../data:/data` (correctly references the root `data/` folder)
  * `../notebooks:/notebooks` (correctly references the root `notebooks/` folder)
* These relative paths resolve correctly because `Streamlit Viewer/` is in the same directory level as the old folder.

### 🧠 Model Loading Integrity
* The Streamlit dashboard pages (`pages/2_Long_Term_Forecasting.py`, `pages/3_Short_Term_Forecasting.py`, and `pages/7_Live_Forecast.py`) load the binary files `ADA_LIVE.joblib` and `CAT_LIVE.cbm` from `../notebooks/`.
* **Important**: We explicitly **excluded** these binary models and the RAG SQLite database (`apple_sales_rag_ollama.db` in `notebooks/`) from renaming. Because their names remain exactly `ADA_LIVE.joblib` and `CAT_LIVE.cbm`, the Streamlit application will load them without error.

---

## ⚠️ 3. Potential Conflicts and Dangers

| Risk Area | Danger | Mitigation Strategy |
| :--- | :--- | :--- |
| **Git Merge Conflicts** | If other collaborators push edits to files under old names (like `walkthrough.md` or `new new target.py`), pulling their changes will create merge conflicts or duplicate files. | Coordinate with collaborators to ensure they are aware of the renaming structure before pulling. |
| **Local Untracked Files** | Running `git clean -fdx` or similar clean commands could wipe out the Power BI dashboards (`*.pbix`) and Excel/CSV files in the root that were gitignored but kept on your local machine. | **Never** run `git clean -fdx` without making an external backup of your `.pbix` files and other untracked local documents. |
| **Hardcoded Script Paths** | Custom scripts you run outside of the standard pipeline may reference the old paths like `app/` or `cleaned_apple_sales_enriched_realistic/` and break. | Update any external script shortcuts or commands to reference `Streamlit Viewer` and `notebooks/017_target_market_validator.py`. |

---

## 🚦 4. Guidelines: Do's and Don'ts

### ✅ What TO DO:
1. **Run Streamlit**: Run the app locally or in Docker from the new directory name:
   ```powershell
   cd "Streamlit Viewer"
   streamlit run app.py
   ```
2. **Execute the Simulation Pipeline**: Run the simulation adjustments script via:
   ```powershell
   python scripts/fix_simulation_factors.py
   ```
3. **Execute Market Expansion Validation**: Run the opportunity scorer via:
   ```powershell
   python notebooks/017_target_market_validator.py
   ```
4. **Refer to the Renaming Map**: Check `010_notebook_renaming_map.md` if you ever get confused about which file corresponds to which phase.

### ❌ What NOT TO DO:
1. **Do NOT rename model binaries**: Never change the filenames of `notebooks/ADA_LIVE.joblib` or `notebooks/CAT_LIVE.cbm`. The Streamlit app looks for these exact strings.
2. **Do NOT delete local untracked files**: Avoid running force-clean commands that delete ignored files, as they will delete your local copies of Power BI dashboards.
3. **Do NOT mix old script runs**: Refrain from trying to invoke deleted scratch files like `check_nb.py` or temporary `.txt` logs.
