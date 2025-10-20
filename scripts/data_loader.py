import pandas as pd
from pathlib import Path

def load_data(base_path='../data/raw'):
    base = Path(base_path)
    category = pd.read_csv(base / 'Category.csv')
    products = pd.read_csv(base / 'Products.csv')
    sales = pd.read_csv(base / 'Sales.csv')
    stores = pd.read_csv(base / 'Stores.csv')
    warranty = pd.read_csv(base / 'Warranty.csv')
    return {
        'category': category,
        'products': products,
        'sales': sales,
        'stores': stores,
        'warranty': warranty
    }
