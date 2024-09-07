import os
import pandas as pd
from utils import load_data, merge_datasets, get_configs

# Load configurations
config = get_configs('configs.yaml')

# Set base path and file paths
base_path = config['data_paths']['base_path']
file_paths = config['data_paths']

# Load data from various sources

df_customers = load_data(os.path.join(base_path, file_paths['customers']), dtype={
    'customer_id': str,
    'address_id': str,
    'birthdate': str,
    'gender': str,
    'job_type': str,
    'email_provider': str,
    'flag_phone_provided': float,
    'flag_privacy': bool
}, parse_dates=['birthdate'])

df_products = load_data(os.path.join(base_path, file_paths['products']), dtype={
    'product_id': str,
    'product_class': str
})

df_orders = load_data(os.path.join(base_path, file_paths['orders']), dtype={
    'order_id': str,
    'customer_id': str,
    'store_id': str,
    'product_id': str,
    'direction': int,
    'gross_price': float,
    'price_reduction': float
}, parse_dates=['purchase_datetime'])

df_addresses = load_data(os.path.join(base_path, file_paths['addresses']), dtype={
    'address_id': str,
    'postal_code': str,
    'district': str,
    'region': str
})

df_customer_accounts = load_data(os.path.join(base_path, file_paths['customer_accounts']), dtype={
    'customer_id': str,
    'account_id': str,
    'favorite_store': str,
    'loyalty_type': str,
    'loyalty_status': bool
}, parse_dates=['activation_date'])

# Merge datasets
customer_data, order_data, merged_data = merge_datasets(df_customers, df_products, df_orders, df_addresses, df_customer_accounts)

# Save the merged data to a CSV file
merged_data.to_csv(os.path.join(base_path, 'merged_data.csv'), index=False)
print(f'[LOG] Merged data saved to {base_path} merged_data.csv')


# Load data for reviews
df_customer_reviews = load_data(os.path.join(base_path, file_paths['customer_reviews']), dtype={
    'review_id': str,
    'customer_id': str,
    'review_text': str
})

df_labelled_reviews = load_data(os.path.join(base_path, file_paths['labelled_reviews']), dtype={
    'labelled_reviews_index': str,
    'review_text': str,
    'sentiment_label': str
})

# Clean the review text
df_customer_reviews['review_text'] = df_customer_reviews['review_text'].str.replace('<', '')
df_customer_reviews['review_text'] = df_customer_reviews['review_text'].str.replace('>', '')
df_customer_reviews['review_text'] = df_customer_reviews['review_text'].str.replace('href', '')
df_customer_reviews['review_text'] = df_customer_reviews['review_text'].str.replace('br', '')

df_labelled_reviews['review_text'] = df_labelled_reviews['review_text'].str.replace('<', '')
df_labelled_reviews['review_text'] = df_labelled_reviews['review_text'].str.replace('>', '')
df_labelled_reviews['review_text'] = df_labelled_reviews['review_text'].str.replace('href', '')
df_labelled_reviews['review_text'] = df_labelled_reviews['review_text'].str.replace('br', '')

save_path = os.path.join(base_path, 'cleaned_reviews.csv')
df_customer_reviews.to_csv(save_path, index=False)
print(f'[LOG] Cleaned reviews saved to {save_path}')

save_path = os.path.join(base_path, 'cleaned_labelled_reviews.csv')
df_labelled_reviews.to_csv(save_path, index=False)
print(f'[LOG] Cleaned labelled reviews saved to {save_path}')