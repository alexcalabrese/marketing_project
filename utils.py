import os
import yaml
import pandas as pd
from IPython.display import display

def get_configs(config_path='/teamspace/studios/this_studio/Marketing Project/configs.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_duplicates(df, cols=None):
    if cols:
        return df[df.duplicated(subset=cols, keep=False)]
    else:
        return df[df.duplicated(keep=False)]

def display_unique_values(df, df_name):
    print(f"\nUnique head values for {df_name}:")
    for col in df.columns:
        print(f"{col}:")
        display(df[col].drop_duplicates().head())

def load_data(file_path, dtype=None, parse_dates=None):
    df = pd.read_csv(file_path, dtype=dtype, parse_dates=parse_dates)
    return df

def merge_datasets(df_customers, df_products, df_orders, df_addresses, df_customer_accounts):
    
    # Merge Customers with Addresses and Accounts
    customer_data = df_customers.merge(df_addresses, on='address_id', how='left')
    customer_data = customer_data.merge(df_customer_accounts, on='customer_id', how='left')

    # Merge Order with Products
    order_data = df_orders.merge(df_products, on='product_id', how='left')

    # Final Merge orders data with customers data    
    merged_data = order_data.merge(customer_data, on='customer_id', how='left')
    
    # Ensure there are no duplicate entries after merge
    merged_data.drop_duplicates(subset=['order_id', 'product_id'], keep='first', inplace=True)

    return customer_data, order_data, merged_data