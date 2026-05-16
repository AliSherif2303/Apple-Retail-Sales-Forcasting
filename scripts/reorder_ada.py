import json

def reorder_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb['cells']
    
    # Identify specific cells by their source content
    train_idx = -1
    for i, c in enumerate(cells):
        src = "".join(c.get('source', []))
        if "18. TRAIN ADABOOST" in src:
            train_idx = i
            break
            
    print(f"Train cell found at index {train_idx}")
    
    # We want the order to be:
    # 0 to 7 (Setup, Preproc, Baseline)
    # Then TRAIN ADABOOST (which is currently 12)
    # Then Markdown Baseline vs AdaBoost (8)
    # Then Comparison (9)
    # Then PLOTS (10)
    # Then 13 to end
    
    new_cells = []
    # 0 to 7
    new_cells.extend(cells[0:8])
    # Train AdaBoost
    new_cells.append(cells[12])
    # Markdown comparison
    new_cells.append(cells[8])
    
    # Comparison code - we will use cell 9, but make sure best_ada = model is there and ada_test_preds = model.predict is used
    comp_cell = cells[9]
    src = "".join(comp_cell.get('source', []))
    if "best_ada = model" not in src:
        # replace the import section
        src = src.replace("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n", 
                          "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n# Map best_ada to the trained model\nbest_ada = model\n")
    
    # just to ensure it's clean:
    src = src.replace("best_ada.predict", "model.predict")
    comp_cell['source'] = [line + "\n" if not line.endswith("\n") else line for line in src.split("\n")]
    if comp_cell['source'] and comp_cell['source'][-1] == "\n":
        comp_cell['source'] = comp_cell['source'][:-1]
    new_cells.append(comp_cell)
    
    # PLOTS
    new_cells.append(cells[10])
    
    # 13 onwards
    new_cells.extend(cells[13:])
    
    nb['cells'] = new_cells
    print(f"New cell count: {len(new_cells)}")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

reorder_notebook("notebooks/ADA_LIVE.ipynb")
