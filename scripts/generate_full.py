import pandas as pd
import numpy as np

# Paths
raw_dir = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\data\raw"
processed_dir = r"c:\Users\Ali Sherif\Apple-Retail-Sales-Forcasting\data\processed"

# 1. Load Data
year_data = pd.read_csv(f"{raw_dir}\\Year-Category-Product-StartingPrice.csv")
category = pd.read_csv(f"{raw_dir}\\category.csv")
new_services = pd.read_csv(f"{raw_dir}\\Productname-Launchdate-Price.csv")

# 2. Process year_data
year_data['Year'] = year_data['Year'].ffill()
df = year_data.merge(category, left_on='Category', right_on='category_name', how='left')

out_df = pd.DataFrame()
out_df['Product_Name'] = df['Product']
out_df['Category_ID'] = df['category_id']
df['Year'] = df['Year'].astype(int).astype(str)
out_df['Launch_Date'] = pd.to_datetime(df['Year'] + '-01-01').dt.strftime('%Y-%m-%d')
out_df['Price'] = df['Starting Price ($)']

# 3. Drop "iPad (Not released)"
out_df = out_df[out_df['Product_Name'] != 'iPad (Not released)'].reset_index(drop=True)

# 4. Generate Product_IDs for the main items
out_df['Product_ID'] = ['P-' + str(i+1) for i in range(len(out_df))]

# 5. Process new services
new_services['Price'] = new_services['Price'].str.replace('$', '', regex=False).str.replace('+', '', regex=False).astype(float)
new_services['Category_ID'] = 'CAT-8'
new_services = new_services.rename(columns={'Product_name': 'Product_Name', 'Launch_date': 'Launch_Date'})

# Generate Product_IDs for the new services
last_id = len(out_df)
new_services['Product_ID'] = ['P-' + str(last_id + i + 1) for i in range(len(new_services))]

# Reorder
new_services = new_services[['Product_ID', 'Product_Name', 'Category_ID', 'Launch_Date', 'Price']]
out_df = out_df[['Product_ID', 'Product_Name', 'Category_ID', 'Launch_Date', 'Price']]

# 6. Combine and save
df_combined = pd.concat([out_df, new_services], ignore_index=True)
df_combined.to_csv(f"{processed_dir}\\generated_products.csv", index=False)

print(f"Success! Generated {len(df_combined)} total products (including services) and removed the unreleased iPad.")
