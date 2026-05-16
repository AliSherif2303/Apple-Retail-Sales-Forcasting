import re, json

path = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\02_exploratory_data_analysis.ipynb"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

count = len(re.findall(r"<{7}", content))
print(f"Conflicts found: {count}")

content = re.sub(
    r"<{7}[^\n]*\n(.*?)={7}\n(.*?)>{7}[^\n]*\n",
    r"\2",
    content,
    flags=re.DOTALL,
)

nb = json.loads(content)
print(f"Valid JSON with {len(nb['cells'])} cells")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed!")
