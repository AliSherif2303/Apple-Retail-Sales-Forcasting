import json

with open(r'C:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\002_preprocess_externals.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(r'C:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\nb_summary.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb.get('cells', [])):
        source = "".join(cell.get('source', []))
        if not source.strip(): continue
        
        if cell['cell_type'] == 'markdown':
            out.write(f"--- MARKDOWN CELL {i} ---\n{source}\n\n")
        elif cell['cell_type'] == 'code':
            out.write(f"--- CODE CELL {i} ---\n{source}\n\n")
