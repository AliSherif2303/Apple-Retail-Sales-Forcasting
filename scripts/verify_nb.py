import json
nb = json.load(open('notebooks/CATBOOST_PROPHET.ipynb', 'r', encoding='utf-8'))
print(f"Notebook format: v{nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")
code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')
print(f"Code cells: {code_cells}  |  Markdown cells: {md_cells}")
print()
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    src = ''.join(c['source'])
    preview = src[:90].replace('\n', ' ')
    print(f"  [{i:2d}] {ctype:8s} | {preview}...")
