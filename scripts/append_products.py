import pandas as pd

# Paths
generated_products_path = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\data\processed\generated_products.csv"
new_products_path = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\data\raw\Productname-Launchdate-Price.csv"

# Load data
df_main = pd.read_csv(generated_products_path)
df_new = pd.read_csv(new_products_path)

# Clean Price column in new data
df_new['Price'] = df_new['Price'].str.replace('$', '', regex=False).str.replace('+', '', regex=False).astype(float)

# Add missing columns
# All these new products appear to be Subscription Services (CAT-8)
df_new['Category_ID'] = 'CAT-8'

# Rename columns to match df_main
df_new = df_new.rename(columns={
    'Product_name': 'Product_Name',
    'Launch_date': 'Launch_Date'
})

# Generate Product_IDs starting from where df_main left off
last_id_num = int(df_main['Product_ID'].iloc[-1].split('-')[1])
df_new['Product_ID'] = ['P-' + str(last_id_num + i + 1) for i in range(len(df_new))]

# Reorder columns to match df_main
df_new = df_new[['Product_ID', 'Product_Name', 'Category_ID', 'Launch_Date', 'Price']]

# Append and save
df_combined = pd.concat([df_main, df_new], ignore_index=True)
df_combined.to_csv(generated_products_path, index=False)

print(f"Successfully appended {len(df_new)} rows to generated_products.csv")
