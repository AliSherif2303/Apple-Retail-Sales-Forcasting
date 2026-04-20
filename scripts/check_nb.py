import json
nb = json.load(open('notebooks/COMBINED_ADABOOST_CATBOOST.ipynb', encoding='utf-8'))
lines = []
lines.append(f"Total cells: {len(nb['cells'])}")
lines.append(f"Code: {sum(1 for c in nb['cells'] if c['cell_type']=='code')}")
lines.append(f"Markdown: {sum(1 for c in nb['cells'] if c['cell_type']=='markdown')}")
lines.append("")
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown' and c['source']:
        first = c['source'][0].strip()[:90]
        # remove non-ascii
        first = first.encode('ascii', 'replace').decode('ascii')
        lines.append(f"  [{i:2d}] {first}")
with open('scripts/nb_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print("Done")
