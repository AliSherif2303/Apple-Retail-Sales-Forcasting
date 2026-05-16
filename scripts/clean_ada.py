import json

def clean_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    print(f"Original cells: {len(nb['cells'])}")
    
    unique_cells = []
    seen_source = set()
    
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        
        # We need to fix the best_ada variable to just use model
        if "best_ada.predict" in src:
            src = src.replace("best_ada.predict", "model.predict")
            cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in src.split("\n")]
            # remove empty trailing lines from split
            if cell['source'] and cell['source'][-1] == "\n":
                cell['source'] = cell['source'][:-1]
                
        # To identify duplicate cells, we use the normalized source
        norm_src = src.strip()
        
        # If it's a plot or comparison code, they are duplicated a lot.
        if norm_src in seen_source and norm_src != "":
            print(f"Removing duplicate cell {i}")
            continue
            
        seen_source.add(norm_src)
        unique_cells.append(cell)
        
    nb['cells'] = unique_cells
    print(f"New cells: {len(nb['cells'])}")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
        
clean_notebook("notebooks/ADA_LIVE.ipynb")
