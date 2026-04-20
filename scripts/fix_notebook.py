"""
Script to fix the COMBINED_ADABOOST_CATBOOST.ipynb notebook:
1. Fix the missing 'num_categories' feature by computing it from 'category_id'
2. Add value labels to horizontal bar visualizations
"""
import json
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "COMBINED_ADABOOST_CATBOOST.ipynb"

def load_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_notebook(nb, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  [OK] Saved {path}")

def join_source(cell):
    """Join cell source lines into a single string."""
    return "".join(cell.get("source", []))

def set_source(cell, text):
    """Set cell source from a single string (split into lines for JSON)."""
    result = []
    all_lines = text.split("\n")
    for i, line in enumerate(all_lines):
        if i < len(all_lines) - 1:
            result.append(line + "\n")
        else:
            if line:  # don't add empty trailing line
                result.append(line)
    cell["source"] = result

def fix_num_categories(nb):
    """
    Fix 1: Add num_categories computation from category_id during aggregation.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = join_source(cell)
        
        if "if 'num_categories' in df_raw.columns:" in src and "monthly_aggs['num_categories'] = 'max'" in src:
            # Add computation of num_categories from category_id when it doesn't exist
            old = "if 'num_categories' in df_raw.columns:\n    monthly_aggs['num_categories'] = 'max'"
            new = ("if 'num_categories' in df_raw.columns:\n"
                   "    monthly_aggs['num_categories'] = 'max'\n"
                   "if 'category_id' in df_raw.columns and 'num_categories' not in df_raw.columns:\n"
                   "    monthly_aggs['category_id'] = 'nunique'")
            src = src.replace(old, new)
            print("  [OK] Added category_id -> num_categories aggregation rule")
            
            # Also add the rename step after the num_unique_products block
            old2 = ("# Add num_unique_products if missing\n"
                    "if 'num_unique_products' not in df_monthly.columns:\n"
                    "    df_monthly['num_unique_products'] = (\n"
                    "        df_raw.groupby(['store_id','year','month'])['product_id']\n"
                    "        .nunique().values\n"
                    "    )")
            new2 = (old2 + "\n\n"
                    "# Add num_categories from category_id if it was aggregated\n"
                    "if 'category_id' in df_monthly.columns and 'num_categories' not in df_monthly.columns:\n"
                    "    df_monthly.rename(columns={'category_id': 'num_categories'}, inplace=True)")
            src = src.replace(old2, new2)
            print("  [OK] Added category_id -> num_categories rename step")
            
            set_source(cell, src)
            return True
    
    print("  [WARN] Could not find preprocessing cell to fix num_categories")
    return False

def fix_feature_importance_bars(nb):
    """
    Fix 2a: Add value labels to the feature importances horizontal bar chart.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = join_source(cell)
        
        if "def plot_model_results(" in src and "imp.nlargest(15).sort_values().plot.barh" in src:
            old = "    imp.nlargest(15).sort_values().plot.barh(ax=ax, color=sns.color_palette('viridis', 15))\n    ax.set_title('Top 15 Feature Importances')"
            new = ("    bars = imp.nlargest(15).sort_values()\n"
                   "    bars.plot.barh(ax=ax, color=sns.color_palette('viridis', 15))\n"
                   "    for bar in ax.patches:\n"
                   "        w = bar.get_width()\n"
                   "        ax.text(w + w*0.01, bar.get_y() + bar.get_height()/2, f'{w:,.4f}', va='center', fontsize=8)\n"
                   "    plt.tight_layout()\n"
                   "    ax.set_title('Top 15 Feature Importances')")
            if old in src:
                src = src.replace(old, new)
                set_source(cell, src)
                print("  [OK] Added value labels to Feature Importances bar chart")
                return True
            else:
                print("  [WARN] Could not find exact feature importance bar pattern")
                return False
    
    print("  [WARN] Could not find plot_model_results function")
    return False

def fix_forecast_bars(nb):
    """
    Fix 2b: Add value labels to the top 10 stores horizontal bar chart.
    """
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = join_source(cell)
        
        if "def plot_forecast(" in src and "top_stores.plot.barh(ax=ax" in src:
            # Try to find the pattern - use the exact text from the notebook
            # The em-dash might be different encodings
            target_title = "Top 10 Stores"
            if target_title in src:
                # Extract the exact title line to handle different dash characters
                idx_title = src.index(target_title)
                # Find the full line containing the title
                line_start = src.rfind("\n", 0, idx_title) + 1
                line_end = src.index("\n", idx_title)
                title_line = src[line_start:line_end]
                
                # Find the barh line
                barh_marker = "top_stores.plot.barh(ax=ax, color=sns.color_palette('magma', 10))"
                idx_barh = src.index(barh_marker)
                barh_line_start = src.rfind("\n", 0, idx_barh) + 1
                barh_line_end = src.index("\n", idx_barh)
                barh_line = src[barh_line_start:barh_line_end]
                
                # Find xlabel line
                xlabel_marker = "ax.set_xlabel('Predicted Sales')"
                idx_xlabel = src.index(xlabel_marker)
                xlabel_line_start = src.rfind("\n", 0, idx_xlabel) + 1
                xlabel_line_end = src.index("\n", idx_xlabel)
                xlabel_line = src[xlabel_line_start:xlabel_line_end]
                
                old = barh_line + "\n" + title_line + "\n" + xlabel_line
                new = (barh_line + "\n"
                       "    for bar in ax.patches:\n"
                       "        w = bar.get_width()\n"
                       "        ax.text(w + w*0.01, bar.get_y() + bar.get_height()/2, f'{w:,.0f}', va='center', fontsize=10)\n"
                       "    plt.tight_layout()\n" +
                       title_line + "\n" + xlabel_line)
                
                if old in src:
                    src = src.replace(old, new)
                    set_source(cell, src)
                    print("  [OK] Added value labels to Top 10 Stores forecast bar chart")
                    return True
                else:
                    print("  [WARN] Pattern mismatch in forecast bars")
                    print(f"  DEBUG old repr: {repr(old[:200])}")
                    return False
            else:
                print("  [WARN] Could not find title pattern in forecast function")
                return False
    
    print("  [WARN] Could not find plot_forecast function")
    return False

def main():
    print(f"Loading notebook: {NOTEBOOK_PATH}")
    nb = load_notebook(NOTEBOOK_PATH)
    
    print("\n--- Fix 1: Add num_categories feature ---")
    fix_num_categories(nb)
    
    print("\n--- Fix 2a: Add value labels to feature importance bars ---")
    fix_feature_importance_bars(nb)
    
    print("\n--- Fix 2b: Add value labels to forecast top-10 stores bars ---")
    fix_forecast_bars(nb)
    
    print("\n--- Saving notebook ---")
    save_notebook(nb, NOTEBOOK_PATH)
    print("\nDone! All fixes applied.")

if __name__ == "__main__":
    main()
