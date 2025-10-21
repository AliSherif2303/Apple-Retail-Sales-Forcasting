import pandas as pd

def merge_all(datasets: dict):
    sales = datasets['sales']
    products = datasets['products']
    stores = datasets['stores']
    category = datasets['category']
    warranty = datasets['warranty']

    df = (
        sales
        .merge(products, left_on='product_id', right_on='Product_ID', how='left')
        .merge(stores, left_on='store_id', right_on='Store_ID', how='left')
        .merge(category, left_on='Category_ID', right_on='category_id', how='left')
        .merge(warranty, on='sale_id', how='left')
    )
    
    return df
