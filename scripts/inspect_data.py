import json
nb = json.load(open('notebooks/ADABOOST.ipynb', 'r', encoding='utf-8'))
print(f"Notebook format: v{nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    src = ''.join(c['source'])
    preview = src[:80].replace('\n', ' ')
    print(f"  [{i:2d}] {ctype:8s} | {preview}...")
