import json

for nb_name in ["CAT_LIVE.ipynb", "CATBOOST_regulized.ipynb", "cat&ada.ipynb"]:
    path = rf"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\{nb_name}"
    try:
        nb = json.load(open(path, "r", encoding="utf-8"))
    except:
        print(f"--- SKIPPED {nb_name} (can't read) ---")
        continue
    print(f"\n=== {nb_name} ===")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        keywords = ["X_train", "train_mask", "test_mask", "holdout", "TimeSeriesSplit", "cutoff", "split"]
        if any(kw in src for kw in keywords):
            print(f"\n--- Cell {i} ({c['cell_type']}) ---")
            print(src[:500])
