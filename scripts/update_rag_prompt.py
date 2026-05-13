import json

# Define the new rules to merge in
new_rules_text = """    IMPORTANT APPLE RETAIL BUSINESS RULES:
    1. If asked about "Sales", "Revenue", or "Income", ALWAYS default to using the 'sales_amount_realistic' column, NOT the base 'sales_amount' column.
    2. If asked about Volume or Item Counts, ALWAYS use 'quantity_realistic'.
    3. If asked about a Country, ALWAYS filter and select using 'country_norm_mapped'. Never put a country in the 'city' column.
    4. There is ONLY ONE table named 'sales'. DO NOT try to JOIN other tables like 'categories'. All category data is already inside the 'sales' table (e.g. 'category_name').
    5. If asked to count "transactions" or "orders", use COUNT(*), not SUM(quantity).
    6. If comparing metrics across different years side-by-side, use conditional aggregation (e.g. `SUM(CASE WHEN year=2023 THEN sales_amount_realistic END)`).
    7. Avoid markdown wrapping. Output only the raw, executable SQL string.
    8. CRITICAL: To find the "most" or "least" (like most expensive item), DO NOT use MAX() or MIN() functions! Instead, just select the raw columns and use `ORDER BY column_name DESC LIMIT 1`.
    9. When asked for the price of a product, always SELECT 'price_realistic', NEVER 'sales_amount_realistic'.
    10. When filtering for specific product lines (like 'MacBook' or 'iPhone'), filter using `product_name LIKE '%MacBook%'` instead of `category_name = 'MacBook'`.
"""

file_path = "notebooks/Ollama_SQL_RAG_Agent.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        source = cell.get("source", [])
        updated_source = []
        is_in_rules_block = False
        made_change = False
        
        for line in source:
            if "IMPORTANT APPLE RETAIL BUSINESS RULES:" in line:
                is_in_rules_block = True
                updated_source.append(new_rules_text)
                made_change = True
            elif is_in_rules_block:
                # Skip the previous rules lines until we hit the end of the prompt formatting block
                if "Only use the following tables:" in line:
                    is_in_rules_block = False
                    updated_source.append(line)
            else:
                updated_source.append(line)
                
        if made_change:
            cell["source"] = updated_source
            print("Prompt updated!")

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    
print("Notebook saved successfully.")
