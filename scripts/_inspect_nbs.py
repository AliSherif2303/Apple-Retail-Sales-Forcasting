import json

for nb_name in ["CAT_LIVE.ipynb", "ADA_LIVE.ipynb"]:
    path = rf"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\{nb_name}"
    nb = json.load(open(path, "r", encoding="utf-8"))
    print(f"\n{'='*60}")
    print(f"  {nb_name} — {len(nb['cells'])} cells")
    print(f"{'='*60}")
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        preview = src[:120].replace("\n", " | ")
        print(f"  [{i:2d}] {c['cell_type']:8s} | {preview}")
