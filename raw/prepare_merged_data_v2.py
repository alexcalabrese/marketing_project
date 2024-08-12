import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# These imports might need adjustment based on your project structure
from utils import get_configs, get_duplicates, load_data, display_unique_values, merge_datasets
from params import REFERENCE_DATE, DEFAULT_JOB_TYPE, DEFAULT_EMAIL_PROVIDER, MISSING_FLOAT_DEFAULT

def prepare_merged_data():
    # Load configurations
    config = get_configs()
    base_path = config['data_paths']['base_path']
    file_paths = config['data_paths']

    # Load datasets
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

    df_labelled_reviews = load_data(os.path.join(base_path, file_paths['labelled_reviews']), dtype={
        'labelled_reviews_index': str,
        'review_text': str,
        'sentiment_label': str
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

    df_customer_reviews = load_data(os.path.join(base_path, file_paths['customer_reviews']), dtype={
        'review_id': str,
        'customer_id': str,
        'review_text': str
    })

    df_customer_accounts = load_data(os.path.join(base_path, file_paths['customer_accounts']), dtype={
        'customer_id': str,
        'account_id': str,
        'favorite_store': str,
        'loyalty_type': str,
        'loyalty_status': bool
    }, parse_dates=['activation_date'])

    # Data preprocessing
    df_customers['job_type'].fillna(DEFAULT_JOB_TYPE, inplace=True)
    df_customers['email_provider'].fillna(DEFAULT_EMAIL_PROVIDER, inplace=True)
    df_customers['flag_phone_provided'].fillna(MISSING_FLOAT_DEFAULT, inplace=True)
    df_customers['flag_phone_provided'] = df_customers['flag_phone_provided'].apply(lambda x: x == 1.0)

    df_orders.drop_duplicates(inplace=True)
    df_addresses.drop_duplicates(inplace=True)

    # Merge datasets
    customer_data, order_data, merged_data = merge_datasets(df_customers, df_products, df_orders, df_addresses, df_customer_accounts)

    # Add sentiment scores to merged data
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df_labelled_reviews['sentiment_score'] = df_labelled_reviews['sentiment_label'].map(sentiment_map)

    # Check if 'review_text' exists in df_labelled_reviews
    if 'review_text' in df_labelled_reviews.columns and 'review_text' in merged_data.columns:
        merged_data = merged_data.merge(df_labelled_reviews[['review_text', 'sentiment_score']], on='review_text', how='left')
    else:
        print("Warning: 'review_text' column not found in df_labelled_reviews. Skipping sentiment score merge.")

    # Add customer reviews
    merged_data = merged_data.merge(df_customer_reviews[['customer_id', 'review_text']], on='customer_id', how='left')

    # Save merged data to CSV
    merged_data.to_csv('merged_data.csv', index=False)
    print("Merged data saved to 'merged_data.csv'")

    return merged_data

if __name__ == "__main__":
    merged_data = prepare_merged_data()
    print(merged_data.info())
    print(merged_data.head())