import json

def analyze_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Total cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        src_preview = src[:50].replace("\n", " ")
        print(f"Cell {i} [{cell['cell_type']}]: {src_preview}")

analyze_notebook("notebooks/ADA_LIVE.ipynb")
