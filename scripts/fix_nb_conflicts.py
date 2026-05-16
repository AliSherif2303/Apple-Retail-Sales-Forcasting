import re, json

path = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\notebooks\01_data_loading_and_merging.ipynb"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

count_before = len(re.findall(r"<{7}", content))
print(f"Conflicts found: {count_before}")

# Keep the REMOTE (incoming) side of every conflict
content = re.sub(
    r"<{7}[^\n]*\n(.*?)={7}\n(.*?)>{7}[^\n]*\n",
    r"\2",
    content,
    flags=re.DOTALL,
)

count_after = len(re.findall(r"<{7}", content))
print(f"Conflicts remaining: {count_after}")

# Validate JSON
try:
    nb = json.loads(content)
    print(f"Valid JSON notebook with {len(nb['cells'])} cells")
except json.JSONDecodeError as e:
    print(f"JSON error: {e}")

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("File saved successfully")
