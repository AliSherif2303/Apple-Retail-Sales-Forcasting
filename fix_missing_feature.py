import json
import os

notebook_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "notebooks",
    "COMBINED_ADABOOST_CATBOOST.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

search_str = """# Add num_unique_products if missing
if 'num_unique_products' not in df_monthly.columns:
    df_monthly['num_unique_products'] = (
        df_raw.groupby(['store_id','year','month'])['product_id']
        .nunique().values
    )"""

replace_str = """# Add num_unique_products if missing
if 'num_unique_products' not in df_monthly.columns:
    df_monthly['num_unique_products'] = (
        df_raw.groupby(['store_id','year','month'])['product_id']
        .nunique().values
    )

# Add num_categories if missing
if 'num_categories' not in df_monthly.columns and 'category_id' in df_raw.columns:
    df_monthly['num_categories'] = (
        df_raw.groupby(['store_id','year','month'])['category_id']
        .nunique().values
    )"""

found = False
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        if search_str in source:
            cell["source"] = [source.replace(search_str, replace_str)]
            found = True
            break

if found:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("OK: Replaced and saved.")
else:
    print("ERROR: Could not find target string.")
