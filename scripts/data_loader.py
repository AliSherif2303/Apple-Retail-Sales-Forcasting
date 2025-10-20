import pandas as pd

def load_data(base_url):
    data = {
        "products": pd.read_csv(base_url + "Products.csv"),
        "sales": pd.read_csv(base_url + "Sales.csv"),
        "warranty": pd.read_csv(base_url + "Warranty.csv"),
        "stores": pd.read_csv(base_url + "Stores.csv"),
        "category": pd.read_csv(base_url + "Category.csv")
    }
    return data
