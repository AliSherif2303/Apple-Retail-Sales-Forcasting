import json

with open("notebooks/COMBINED_ADABOOST_CATBOOST.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "monthly_aggs" in src and "num_categories" in src:
        # Check if the fix is present
        has_fix1 = "category_id" in src and "nunique" in src
        has_fix2 = "rename" in src and "num_categories" in src
        print(f"Cell {i}: Fix1 (aggregation rule) = {has_fix1}, Fix2 (rename) = {has_fix2}")
        
        # Print just the relevant lines
        for line in src.split("\n"):
            if "category_id" in line or "num_categories" in line or "rename" in line:
                print(f"  >> {line}")
        break
