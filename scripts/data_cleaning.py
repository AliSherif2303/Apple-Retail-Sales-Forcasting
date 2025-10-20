import pandas as pd

def merge_data(d):
    merged = (
        d["sales"]
        .merge(d["products"], left_on="product_id", right_on="Product_ID", how="left")
        .merge(d["stores"], left_on="store_id", right_on="Store_ID", how="left")
        .merge(d["category"], left_on="Category_ID", right_on="category_id", how="left")
        .merge(d["warranty"], on="sale_id", how="left")
    )
    return merged
