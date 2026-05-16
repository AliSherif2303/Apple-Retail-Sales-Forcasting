"""
inject_baselines.py
====================
Injects baseline model cells into CAT_LIVE.ipynb and ADA_LIVE.ipynb
WITHOUT modifying any existing cells.

- CAT_LIVE: Linear Regression baseline (inserted before CatBoost training)
- ADA_LIVE: Decision Tree Regressor baseline (inserted before AdaBoost training)
"""
import json
import copy

NB_DIR = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks"

def make_md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split("\n")}

def make_code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.split("\n"),
    }


# ═══════════════════════════════════════════════════════════════
# CAT_LIVE.ipynb — Linear Regression Baseline
# ═══════════════════════════════════════════════════════════════

cat_md = make_md(
    "## Baseline Model — Linear Regression\n"
    "\n"
    "Before training CatBoost, we establish a **Linear Regression baseline** using the exact same features and train/test split.\n"
    "This provides a lower-bound reference to quantify how much CatBoost improves over a simple linear model."
)

cat_baseline_code = make_code(
    "# =============================================================================\n"
    "# BASELINE: LINEAR REGRESSION\n"
    "# =============================================================================\n"
    "\n"
    "from sklearn.linear_model import LinearRegression\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "import numpy as np\n"
    "\n"
    "# Train baseline on the SAME features & split\n"
    "lr_baseline = LinearRegression()\n"
    "lr_baseline.fit(X_train, y_train)\n"
    "\n"
    "# Predictions\n"
    "lr_train_preds = lr_baseline.predict(X_train)\n"
    "lr_test_preds  = lr_baseline.predict(X_test)\n"
    "\n"
    "# Metrics\n"
    "lr_train_mae  = mean_absolute_error(y_train, lr_train_preds)\n"
    "lr_train_rmse = np.sqrt(mean_squared_error(y_train, lr_train_preds))\n"
    "lr_train_r2   = r2_score(y_train, lr_train_preds)\n"
    "\n"
    "lr_test_mae   = mean_absolute_error(y_test, lr_test_preds)\n"
    "lr_test_rmse  = np.sqrt(mean_squared_error(y_test, lr_test_preds))\n"
    "lr_test_r2    = r2_score(y_test, lr_test_preds)\n"
    "\n"
    "print('=' * 60)\n"
    "print('BASELINE: Linear Regression')\n"
    "print('=' * 60)\n"
    "print(f'  Train MAE:  ${lr_train_mae:,.2f}')\n"
    "print(f'  Train RMSE: ${lr_train_rmse:,.2f}')\n"
    "print(f'  Train R²:   {lr_train_r2:.4f}')\n"
    "print()\n"
    "print(f'  Test MAE:   ${lr_test_mae:,.2f}')\n"
    "print(f'  Test RMSE:  ${lr_test_rmse:,.2f}')\n"
    "print(f'  Test R²:    {lr_test_r2:.4f}')\n"
    "print('=' * 60)\n"
)

cat_compare_md = make_md(
    "## Baseline vs CatBoost — Comparison\n"
    "\n"
    "The cells below compare the Linear Regression baseline against CatBoost on the test set.\n"
    "Run these **after** the CatBoost training cell has finished."
)

cat_compare_code = make_code(
    "# =============================================================================\n"
    "# COMPARISON: Linear Regression Baseline vs CatBoost\n"
    "# =============================================================================\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "\n"
    "# --- CatBoost metrics (from the training cell above) ---\n"
    "cat_test_preds = model.predict(X_test)\n"
    "cat_train_preds = model.predict(X_train)\n"
    "\n"
    "cat_test_mae  = mean_absolute_error(y_test, cat_test_preds)\n"
    "cat_test_rmse = np.sqrt(mean_squared_error(y_test, cat_test_preds))\n"
    "cat_test_r2   = r2_score(y_test, cat_test_preds)\n"
    "cat_train_r2  = r2_score(y_train, cat_train_preds)\n"
    "\n"
    "# --- Print comparison table ---\n"
    "print('=' * 65)\n"
    "print(f'{\"Metric\":<20} {\"Linear Regression\":>20} {\"CatBoost\":>20}')\n"
    "print('=' * 65)\n"
    "print(f'{\"Test MAE\":<20} {f\"${lr_test_mae:,.0f}\":>20} {f\"${cat_test_mae:,.0f}\":>20}')\n"
    "print(f'{\"Test RMSE\":<20} {f\"${lr_test_rmse:,.0f}\":>20} {f\"${cat_test_rmse:,.0f}\":>20}')\n"
    "print(f'{\"Test R²\":<20} {f\"{lr_test_r2:.4f}\":>20} {f\"{cat_test_r2:.4f}\":>20}')\n"
    "print(f'{\"Train R²\":<20} {f\"{lr_train_r2:.4f}\":>20} {f\"{cat_train_r2:.4f}\":>20}')\n"
    "print('=' * 65)\n"
    "\n"
    "mae_improve = (lr_test_mae - cat_test_mae) / lr_test_mae * 100\n"
    "r2_improve  = (cat_test_r2 - lr_test_r2) / (1 - lr_test_r2) * 100 if lr_test_r2 < 1 else 0\n"
    "print(f'\\nCatBoost reduces MAE by {mae_improve:.1f}% vs Linear Regression baseline.')\n"
    "print(f'CatBoost closes {r2_improve:.1f}% of the remaining R² gap.')\n"
)

cat_plot_code = make_code(
    "# =============================================================================\n"
    "# PLOTS: Baseline vs CatBoost\n"
    "# =============================================================================\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
    "fig.suptitle('Linear Regression Baseline vs CatBoost', fontsize=16, fontweight='bold', y=1.01)\n"
    "\n"
    "# --- 1. Bar chart: MAE & RMSE comparison ---\n"
    "ax = axes[0, 0]\n"
    "metrics = ['MAE', 'RMSE']\n"
    "lr_vals = [lr_test_mae, lr_test_rmse]\n"
    "cat_vals = [cat_test_mae, cat_test_rmse]\n"
    "x = np.arange(len(metrics))\n"
    "w = 0.35\n"
    "bars1 = ax.bar(x - w/2, lr_vals, w, label='Linear Regression', color='#94a3b8', edgecolor='white')\n"
    "bars2 = ax.bar(x + w/2, cat_vals, w, label='CatBoost', color='#818cf8', edgecolor='white')\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels(metrics)\n"
    "ax.set_ylabel('Error ($)')\n"
    "ax.set_title('Test Set Error Comparison')\n"
    "ax.legend()\n"
    "for bar in bars1:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)\n"
    "for bar in bars2:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)\n"
    "\n"
    "# --- 2. Bar chart: R² comparison ---\n"
    "ax = axes[0, 1]\n"
    "r2_labels = ['Train R²', 'Test R²']\n"
    "lr_r2 = [lr_train_r2, lr_test_r2]\n"
    "cat_r2 = [cat_train_r2, cat_test_r2]\n"
    "bars1 = ax.bar(np.arange(2) - w/2, lr_r2, w, label='Linear Regression', color='#94a3b8', edgecolor='white')\n"
    "bars2 = ax.bar(np.arange(2) + w/2, cat_r2, w, label='CatBoost', color='#818cf8', edgecolor='white')\n"
    "ax.set_xticks(np.arange(2))\n"
    "ax.set_xticklabels(r2_labels)\n"
    "ax.set_ylabel('R² Score')\n"
    "ax.set_title('R² Comparison')\n"
    "ax.set_ylim(0, 1.05)\n"
    "ax.legend()\n"
    "for bar in bars1:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)\n"
    "for bar in bars2:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)\n"
    "\n"
    "# --- 3. Scatter: Baseline Actual vs Predicted ---\n"
    "ax = axes[1, 0]\n"
    "ax.scatter(y_test, lr_test_preds, alpha=0.3, s=10, color='#94a3b8')\n"
    "mn, mx = y_test.min(), y_test.max()\n"
    "ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')\n"
    "ax.set_xlabel('Actual Sales ($)')\n"
    "ax.set_ylabel('Predicted Sales ($)')\n"
    "ax.set_title(f'Linear Regression — Actual vs Predicted (R²={lr_test_r2:.4f})')\n"
    "ax.legend()\n"
    "\n"
    "# --- 4. Scatter: CatBoost Actual vs Predicted ---\n"
    "ax = axes[1, 1]\n"
    "ax.scatter(y_test, cat_test_preds, alpha=0.3, s=10, color='#818cf8')\n"
    "ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')\n"
    "ax.set_xlabel('Actual Sales ($)')\n"
    "ax.set_ylabel('Predicted Sales ($)')\n"
    "ax.set_title(f'CatBoost — Actual vs Predicted (R²={cat_test_r2:.4f})')\n"
    "ax.legend()\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)

# ═══════════════════════════════════════════════════════════════
# ADA_LIVE.ipynb — Decision Tree Regressor Baseline
# ═══════════════════════════════════════════════════════════════

ada_md = make_md(
    "## Baseline Model — Decision Tree Regressor\n"
    "\n"
    "Before training AdaBoost, we establish a **Decision Tree Regressor baseline** using the exact same features and train/test split.\n"
    "This provides a lower-bound reference to quantify how much the AdaBoost ensemble improves over a single decision tree."
)

ada_baseline_code = make_code(
    "# =============================================================================\n"
    "# BASELINE: DECISION TREE REGRESSOR\n"
    "# =============================================================================\n"
    "\n"
    "from sklearn.tree import DecisionTreeRegressor\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "import numpy as np\n"
    "\n"
    "# Train baseline on the SAME features & split\n"
    "dt_baseline = DecisionTreeRegressor(max_depth=6, random_state=42)\n"
    "dt_baseline.fit(X_train, y_train)\n"
    "\n"
    "# Predictions\n"
    "dt_train_preds = dt_baseline.predict(X_train)\n"
    "dt_test_preds  = dt_baseline.predict(X_test)\n"
    "\n"
    "# Metrics\n"
    "dt_train_mae  = mean_absolute_error(y_train, dt_train_preds)\n"
    "dt_train_rmse = np.sqrt(mean_squared_error(y_train, dt_train_preds))\n"
    "dt_train_r2   = r2_score(y_train, dt_train_preds)\n"
    "\n"
    "dt_test_mae   = mean_absolute_error(y_test, dt_test_preds)\n"
    "dt_test_rmse  = np.sqrt(mean_squared_error(y_test, dt_test_preds))\n"
    "dt_test_r2    = r2_score(y_test, dt_test_preds)\n"
    "\n"
    "print('=' * 60)\n"
    "print('BASELINE: Decision Tree Regressor (max_depth=6)')\n"
    "print('=' * 60)\n"
    "print(f'  Train MAE:  ${dt_train_mae:,.2f}')\n"
    "print(f'  Train RMSE: ${dt_train_rmse:,.2f}')\n"
    "print(f'  Train R²:   {dt_train_r2:.4f}')\n"
    "print()\n"
    "print(f'  Test MAE:   ${dt_test_mae:,.2f}')\n"
    "print(f'  Test RMSE:  ${dt_test_rmse:,.2f}')\n"
    "print(f'  Test R²:    {dt_test_r2:.4f}')\n"
    "print('=' * 60)\n"
)

ada_compare_md = make_md(
    "## Baseline vs AdaBoost — Comparison\n"
    "\n"
    "The cells below compare the Decision Tree baseline against AdaBoost on the test set.\n"
    "Run these **after** the AdaBoost training cell has finished."
)

ada_compare_code = make_code(
    "# =============================================================================\n"
    "# COMPARISON: Decision Tree Baseline vs AdaBoost\n"
    "# =============================================================================\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
    "\n"
    "# --- AdaBoost metrics (from the training cell above) ---\n"
    "ada_test_preds = best_ada.predict(X_test)\n"
    "ada_train_preds = best_ada.predict(X_train)\n"
    "\n"
    "ada_test_mae  = mean_absolute_error(y_test, ada_test_preds)\n"
    "ada_test_rmse = np.sqrt(mean_squared_error(y_test, ada_test_preds))\n"
    "ada_test_r2   = r2_score(y_test, ada_test_preds)\n"
    "ada_train_r2  = r2_score(y_train, ada_train_preds)\n"
    "\n"
    "# --- Print comparison table ---\n"
    "print('=' * 65)\n"
    "print(f'{\"Metric\":<20} {\"Decision Tree\":>20} {\"AdaBoost\":>20}')\n"
    "print('=' * 65)\n"
    "print(f'{\"Test MAE\":<20} {f\"${dt_test_mae:,.0f}\":>20} {f\"${ada_test_mae:,.0f}\":>20}')\n"
    "print(f'{\"Test RMSE\":<20} {f\"${dt_test_rmse:,.0f}\":>20} {f\"${ada_test_rmse:,.0f}\":>20}')\n"
    "print(f'{\"Test R²\":<20} {f\"{dt_test_r2:.4f}\":>20} {f\"{ada_test_r2:.4f}\":>20}')\n"
    "print(f'{\"Train R²\":<20} {f\"{dt_train_r2:.4f}\":>20} {f\"{ada_train_r2:.4f}\":>20}')\n"
    "print('=' * 65)\n"
    "\n"
    "mae_improve = (dt_test_mae - ada_test_mae) / dt_test_mae * 100\n"
    "r2_improve  = (ada_test_r2 - dt_test_r2) / (1 - dt_test_r2) * 100 if dt_test_r2 < 1 else 0\n"
    "print(f'\\nAdaBoost reduces MAE by {mae_improve:.1f}% vs Decision Tree baseline.')\n"
    "print(f'AdaBoost closes {r2_improve:.1f}% of the remaining R² gap.')\n"
)

ada_plot_code = make_code(
    "# =============================================================================\n"
    "# PLOTS: Baseline vs AdaBoost\n"
    "# =============================================================================\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import numpy as np\n"
    "\n"
    "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n"
    "fig.suptitle('Decision Tree Baseline vs AdaBoost', fontsize=16, fontweight='bold', y=1.01)\n"
    "\n"
    "# --- 1. Bar chart: MAE & RMSE comparison ---\n"
    "ax = axes[0, 0]\n"
    "metrics = ['MAE', 'RMSE']\n"
    "dt_vals = [dt_test_mae, dt_test_rmse]\n"
    "ada_vals = [ada_test_mae, ada_test_rmse]\n"
    "x = np.arange(len(metrics))\n"
    "w = 0.35\n"
    "bars1 = ax.bar(x - w/2, dt_vals, w, label='Decision Tree', color='#94a3b8', edgecolor='white')\n"
    "bars2 = ax.bar(x + w/2, ada_vals, w, label='AdaBoost', color='#818cf8', edgecolor='white')\n"
    "ax.set_xticks(x)\n"
    "ax.set_xticklabels(metrics)\n"
    "ax.set_ylabel('Error ($)')\n"
    "ax.set_title('Test Set Error Comparison')\n"
    "ax.legend()\n"
    "for bar in bars1:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)\n"
    "for bar in bars2:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'${bar.get_height():,.0f}', ha='center', va='bottom', fontsize=9)\n"
    "\n"
    "# --- 2. Bar chart: R² comparison ---\n"
    "ax = axes[0, 1]\n"
    "r2_labels = ['Train R²', 'Test R²']\n"
    "dt_r2 = [dt_train_r2, dt_test_r2]\n"
    "ada_r2 = [ada_train_r2, ada_test_r2]\n"
    "bars1 = ax.bar(np.arange(2) - w/2, dt_r2, w, label='Decision Tree', color='#94a3b8', edgecolor='white')\n"
    "bars2 = ax.bar(np.arange(2) + w/2, ada_r2, w, label='AdaBoost', color='#818cf8', edgecolor='white')\n"
    "ax.set_xticks(np.arange(2))\n"
    "ax.set_xticklabels(r2_labels)\n"
    "ax.set_ylabel('R² Score')\n"
    "ax.set_title('R² Comparison')\n"
    "ax.set_ylim(0, 1.05)\n"
    "ax.legend()\n"
    "for bar in bars1:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)\n"
    "for bar in bars2:\n"
    "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),\n"
    "            f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)\n"
    "\n"
    "# --- 3. Scatter: Baseline Actual vs Predicted ---\n"
    "ax = axes[1, 0]\n"
    "ax.scatter(y_test, dt_test_preds, alpha=0.3, s=10, color='#94a3b8')\n"
    "mn, mx = y_test.min(), y_test.max()\n"
    "ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')\n"
    "ax.set_xlabel('Actual Sales ($)')\n"
    "ax.set_ylabel('Predicted Sales ($)')\n"
    "ax.set_title(f'Decision Tree — Actual vs Predicted (R²={dt_test_r2:.4f})')\n"
    "ax.legend()\n"
    "\n"
    "# --- 4. Scatter: AdaBoost Actual vs Predicted ---\n"
    "ax = axes[1, 1]\n"
    "ax.scatter(y_test, ada_test_preds, alpha=0.3, s=10, color='#818cf8')\n"
    "ax.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect')\n"
    "ax.set_xlabel('Actual Sales ($)')\n"
    "ax.set_ylabel('Predicted Sales ($)')\n"
    "ax.set_title(f'AdaBoost — Actual vs Predicted (R²={ada_test_r2:.4f})')\n"
    "ax.legend()\n"
    "\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
)


# ═══════════════════════════════════════════════════════════════
# INJECT INTO NOTEBOOKS
# ═══════════════════════════════════════════════════════════════

def inject(nb_path, insert_before_idx, new_cells):
    """Insert new_cells before insert_before_idx in the notebook."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    before = nb["cells"][:insert_before_idx]
    after  = nb["cells"][insert_before_idx:]
    nb["cells"] = before + new_cells + after

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"  Injected {len(new_cells)} cells before index {insert_before_idx}")
    print(f"  Total cells: {len(nb['cells'])}")


# --- CAT_LIVE.ipynb ---
cat_path = f"{NB_DIR}\\CAT_LIVE.ipynb"
print(f"\n{'='*60}")
print(f"  Injecting into CAT_LIVE.ipynb")
print(f"{'='*60}")

# Baseline cells go BEFORE the CatBoost training (cell 6)
# Comparison cells go AFTER it (so after cell 6 = after new cell 6+4 = cell 10)
cat_baseline_cells = [cat_md, cat_baseline_code]
inject(cat_path, 6, cat_baseline_cells)

# Now CatBoost training is at index 8. Comparison goes after it (index 9)
with open(cat_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
cat_compare_cells = [cat_compare_md, cat_compare_code, cat_plot_code]
before = nb["cells"][:9]
after  = nb["cells"][9:]
nb["cells"] = before + cat_compare_cells + after
with open(cat_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"  Injected 3 comparison cells after CatBoost training")
print(f"  Final total: {len(nb['cells'])} cells")


# --- ADA_LIVE.ipynb ---
ada_path = f"{NB_DIR}\\ADA_LIVE.ipynb"
print(f"\n{'='*60}")
print(f"  Injecting into ADA_LIVE.ipynb")
print(f"{'='*60}")

# Baseline cells go BEFORE the AdaBoost training (cell 6)
ada_baseline_cells = [ada_md, ada_baseline_code]
inject(ada_path, 6, ada_baseline_cells)

# Now AdaBoost training is at index 8. Comparison goes after it (index 9)
with open(ada_path, "r", encoding="utf-8") as f:
    nb = json.load(f)
ada_compare_cells = [ada_compare_md, ada_compare_code, ada_plot_code]
before = nb["cells"][:9]
after  = nb["cells"][9:]
nb["cells"] = before + ada_compare_cells + after
with open(ada_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"  Injected 3 comparison cells after AdaBoost training")
print(f"  Final total: {len(nb['cells'])} cells")

print("\n[DONE] Both notebooks updated successfully!")
