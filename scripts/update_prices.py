import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import os

print("Starting pricing update...")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT_ROOT / "data" / "processed"
CSV_FILE = PROC_DIR / "cleaned_apple_sales_enriched_realistic.csv"

# 1. Base Price Mapping
base_prices = {
    # iPhones
    'iPhone 12 mini': 699, 'iPhone 12': 829, 'iPhone 12 Pro': 999, 'iPhone 12 Pro Max': 1099,
    'iPhone 13 mini': 699, 'iPhone 13': 799, 'iPhone 13 Pro': 999, 'iPhone 13 Pro Max': 1099,
    'iPhone 14': 799, 'iPhone 14 Plus': 899, 'iPhone 14 Pro': 999, 'iPhone 14 Pro Max': 1099,
    'iPhone SE (3rd Generation)': 429,
    # iPads
    'iPad (9th Generation)': 329, 'iPad (10th Generation)': 449,
    'iPad mini (5th Generation)': 399, 'iPad mini (6th Generation)': 499,
    'iPad Air (4th Generation)': 599, 'iPad Air (5th Generation)': 599,
    'iPad Pro 11-inch': 799, 'iPad Pro 12.9-inch': 1099, 'iPad Pro (M1)': 799, 'iPad Pro (M2)': 799,
    # Macs
    'MacBook Air (Retina)': 999, 'MacBook Air (M1)': 999, 'MacBook Air (M2)': 1199,
    'MacBook Pro 13-inch': 1299, 'MacBook Pro 14-inch': 1999, 'MacBook Pro 16-inch': 2499, 'MacBook Pro (Touch Bar)': 1299,
    'MacBook': 1299, 'MacBook (Retina)': 1299, 'MacBook (Early 2015)': 1299,
    'iMac 24-inch': 1299, 'iMac 27-inch': 1799, 'iMac Pro': 4999, 'iMac with Retina Display': 1499,
    'Mac Mini': 699, 'Mac Mini (M2)': 599, 'Mac Studio': 1999, 'Mac Pro (Tower)': 5999, 'Mac Pro (Rack)': 6499, 'Mac Pro (2023)': 6999,
    # Watches
    'Apple Watch Series 5': 399, 'Apple Watch Series 6': 399, 'Apple Watch Series 7': 399, 'Apple Watch Series 8': 399, 'Apple Watch Series 9': 399,
    'Apple Watch SE': 249, 'Apple Watch Ultra': 799, 'Apple Watch Herms': 1249, 'Apple Watch Hermès': 1249, 'Apple Watch Nike Edition': 399,
    # Audio
    'AirPods (2nd Generation)': 129, 'AirPods (3rd Generation)': 169,
    'AirPods Pro': 249, 'AirPods Pro (2nd Generation)': 249, 'AirPods Max': 549,
    'HomePod mini': 99, 'HomePod': 299, 'HomePod (2nd Generation)': 299,
    'Beats Solo Pro': 299, 'Beats Studio Buds': 149, 'Beats Powerbeats Pro': 249, 'Beats Fit Pro': 199,
    # TV
    'Apple TV (3rd Generation)': 69, 'Apple TV HD': 149, 'Apple TV 4K': 129,
    # Accessories
    'AirTag': 29, 'Magic Mouse': 79, 'Magic Trackpad': 129, 'Magic Keyboard': 99, 'Magic Keyboard with Touch ID': 149,
    'Apple Pencil (1st Generation)': 99, 'Apple Pencil (2nd Generation)': 129,
    'MagSafe Charger': 39, 'MagSafe Battery Pack': 99, 'Lightning to USB Cable': 19,
    'Smart Cover for iPad': 49, 'Smart Keyboard Folio': 179, 'Silicone Case for iPhone': 49, 'Leather Case for iPhone': 59,
    # Services
    'Apple One': 199, 'Apple Fitness+': 79, 'Apple Music': 109, 'Apple News+': 119, 'Apple Arcade': 49, 'Apple TV+': 69, 'iCloud': 35
}

# Determine product categories for depreciation rules
def get_category(name):
    name_lower = name.lower()
    if 'iphone' in name_lower: return 'iphone'
    if 'ipad' in name_lower: return 'ipad'
    if 'mac' in name_lower or 'imac' in name_lower: return 'mac'
    if 'watch' in name_lower: return 'watch'
    if 'airpods' in name_lower or 'beats' in name_lower or 'homepod' in name_lower: return 'audio'
    if 'apple tv' in name_lower and '+' not in name_lower: return 'tv'
    if 'apple one' in name_lower or '+' in name_lower or 'music' in name_lower or 'icloud' in name_lower or 'arcade' in name_lower: return 'service'
    return 'accessory'

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

print("Applying base prices...")
# Apply base prices (handle missing by taking a median or average)
default_price = 199
df['price'] = df['product_name'].map(base_prices)
missing_products = df[df['price'].isna()]['product_name'].unique()
if len(missing_products) > 0:
    print(f"Warning: Products without base price: {missing_products}")
    # Handle the Hermès encoding issue precisely
    df['price'].fillna(default_price, inplace=True)

print("Calculating launch years and ages...")
df['launch_year'] = pd.to_datetime(df['launch_date']).dt.year

print("Applying depreciation logic...")
def calculate_realistic_price(row):
    bp = row['price']
    cat = get_category(row['product_name'])
    year_diff = row['year'] - row['launch_year']
    
    # If sold before launch date (data anomaly), diff is 0
    if year_diff < 0:
        year_diff = 0
        
    # Apply logic
    if cat == 'iphone':
        # Drop roughly $100 per year, but not below 40% of base price
        drop = 100 * year_diff
        return max(bp - drop, bp * 0.4)
    elif cat in ['ipad', 'watch']:
        # Drop roughly 15% per year
        return bp * (1.0 - 0.15)**year_diff
    elif cat == 'mac':
        # Drop roughly 10% per year
        return bp * (1.0 - 0.10)**year_diff
    elif cat in ['audio', 'tv', 'accessory']:
        # Slower drop, maybe 5% per year
        return bp * (1.0 - 0.05)**year_diff
    elif cat == 'service':
        # Services don't decrease in price, sometimes increase, we will hold constant
        return bp
    
    return bp

df['price_realistic'] = df.apply(calculate_realistic_price, axis=1)

print("Recalculating sales amounts...")
df['sales_amount'] = df['quantity'] * df['price']
df['sales_amount_realistic'] = df['quantity_realistic'] * df['price_realistic']

print("Saving updated CSV dataset...")
df.to_csv(CSV_FILE, index=False)

print("Prices updated successfully!")
print("Here is a sample of the new pricing for iPhone 12 over time:")
print(df[df['product_name'] == 'iPhone 12'].groupby('year')['price_realistic'].mean())
