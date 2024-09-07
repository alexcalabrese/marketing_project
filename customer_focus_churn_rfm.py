import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
from typing import Tuple
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path: str = 'merged_data.csv') -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file, by default 'merged_data.csv'.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.
    """
    with open('/teamspace/studios/this_studio/marketing_project/configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    full_path = os.path.join(config['data_paths']['base_path'], path)
    return pd.read_csv(full_path)

def calculate_rfm(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Recency, Frequency, and Monetary (RFM) values for each customer.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing customer transaction data.

    Returns
    -------
    pd.DataFrame
        DataFrame with RFM values for each customer.
    """
    current_date = pd.to_datetime(data['purchase_datetime']).max()
    
    def calculate_months_active(first_purchase: pd.Timestamp, last_purchase: pd.Timestamp) -> int:
        return ((last_purchase.year - first_purchase.year) * 12 + last_purchase.month - first_purchase.month + 1)
    
    rfm = data.groupby('customer_id').agg({
        'purchase_datetime': ['min', 'max', lambda x: (current_date - pd.to_datetime(x.max())).days],
        'order_id': 'count',
        'gross_price': 'sum'
    }).rename(columns={
        'purchase_datetime': 'recency',
        'order_id': 'order_count',
        'gross_price': 'monetary'
    })
    
    rfm.columns = ['first_purchase', 'last_purchase', 'recency', 'order_count', 'monetary']
    
    rfm['first_purchase'] = pd.to_datetime(rfm['first_purchase'])
    rfm['last_purchase'] = pd.to_datetime(rfm['last_purchase'])
    
    rfm['months_active'] = rfm.apply(lambda row: calculate_months_active(row['first_purchase'], row['last_purchase']), axis=1)
    rfm['frequency'] = rfm['order_count'] / rfm['months_active']
    rfm = rfm.drop(columns=['order_count', 'first_purchase', 'last_purchase', 'months_active'])
    rfm.columns = ['recency', 'monetary', 'frequency']
    
    return rfm

def create_sliding_window_churn_label(data, observation_days=90, control_days=30, step_size=30):
    """
    Create churn labels using a sliding window technique with observation and control periods.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing customer purchase data
    observation_days (int): Number of days in the observation period
    control_days (int): Number of days in the control period
    step_size (int): Number of days to move the window in each step
    
    Returns:
    pd.DataFrame: DataFrame with churn labels for each customer at different time points
    """
    # Ensure purchase_datetime is in datetime format
    data['purchase_datetime'] = pd.to_datetime(data['purchase_datetime'])
    
    # Sort data by customer_id and purchase_datetime
    data = data.sort_values(['customer_id', 'purchase_datetime'])
    
    # Get the overall date range
    start_date = data['purchase_datetime'].min()
    end_date = data['purchase_datetime'].max()
    
    # Initialize an empty list to store results
    results = []
    
    # Slide the window
    current_date = start_date
    while current_date <= end_date - timedelta(days=observation_days + control_days):
        observation_start = current_date
        observation_end = observation_start + timedelta(days=observation_days)
        control_end = observation_end + timedelta(days=control_days)
        
        # Filter data for the current observation window
        observation_data = data[(data['purchase_datetime'] >= observation_start) & 
                                (data['purchase_datetime'] < observation_end)]
        
        # Filter data for the current control window
        control_data = data[(data['purchase_datetime'] >= observation_end) & 
                            (data['purchase_datetime'] < control_end)]
        
        # Get unique customers who made a purchase in the observation period
        customers_observation = set(observation_data['customer_id'])
        
        # Get unique customers who made a purchase in the control period
        customers_control = set(control_data['customer_id'])
        
        # Identify churned customers (those in observation but not in control)
        churned_customers = customers_observation - customers_control
        
        # Create a DataFrame with the results
        window_results = pd.DataFrame({
            'customer_id': list(customers_observation),
            'churned': [1 if cust in churned_customers else 0 for cust in customers_observation],
            'window_start': observation_start,
            'window_end': observation_end,
            'control_end': control_end
        })
        
        results.append(window_results)
        
        # Move the window
        current_date += timedelta(days=step_size)
    
    # Combine all results
    churn_labels = pd.concat(results, ignore_index=True)
    
    return churn_labels

def prepare_features(data: pd.DataFrame, churn_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training based on the sliding window churn labels.

    Parameters:
    data (pd.DataFrame): Original purchase data
    churn_labels (pd.DataFrame): Churn labels from sliding window approach

    Returns:
    pd.DataFrame: Prepared features for each customer-window combination
    """
    features = []

    for _, window in churn_labels.groupby(['window_start', 'window_end']):
        window_start = window['window_start'].iloc[0]
        window_end = window['window_end'].iloc[0]
        
        # Filter data for the current window
        window_data = data[(data['purchase_datetime'] >= window_start) & 
                           (data['purchase_datetime'] < window_end)]
        
        # Calculate features
        customer_features = window_data.groupby('customer_id').agg({
            'purchase_datetime': 'count',  # Frequency
            'gross_price': ['sum', 'mean', 'max']  # Monetary
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'frequency', 'total_spend', 'avg_spend', 'max_spend']
        
        # Calculate recency
        last_purchase = window_data.groupby('customer_id')['purchase_datetime'].max().reset_index()
        last_purchase['recency'] = (window_end - last_purchase['purchase_datetime']).dt.days
        
        # Merge features
        customer_features = customer_features.merge(last_purchase[['customer_id', 'recency']], on='customer_id')
        
        # Add window information
        customer_features['window_start'] = window_start
        customer_features['window_end'] = window_end
        
        features.append(customer_features)
    
    features_df = pd.concat(features, ignore_index=True)
    
    # Merge with churn labels
    final_features = features_df.merge(churn_labels, on=['customer_id', 'window_start', 'window_end'])
    
    return final_features

def train_churn_model(dataset: pd.DataFrame) -> Tuple[RandomForestClassifier, list, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Train a Random Forest model to predict churn.

    Parameters:
    dataset (pd.DataFrame): Prepared features and churn labels

    Returns:
    tuple: Trained model, feature names, and train/test split data
    """
    features = ['recency', 'frequency', 'total_spend', 'avg_spend', 'max_spend']
    X = dataset[features]
    y = dataset['churned']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, features, (X_train_scaled, X_test_scaled, y_train, y_test)

def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series) -> None:
    """
    Evaluate the trained model.

    Parameters:
    model: Trained model
    X_test: Test features
    y_test: Test labels
    feature_names: List of feature names
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {roc_auc:.4f}")

    # # Feature importance
    # feature_importance = pd.DataFrame({
    #     'feature': model.feature_names_in_,
    #     'importance': model.feature_importances_
    # }).sort_values('importance', ascending=False)

    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='importance', y='feature', data=feature_importance)
    # plt.title('Feature Importance')
    # plt.tight_layout()
    # plt.show()

    return {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc
        # 'feature_importance': feature_importance
    }

def identify_high_value_customers(data: pd.DataFrame, model, features, scaler, threshold: float = 0.7) -> pd.DataFrame:
    """
    Identify high-value customers at risk of churning.

    Parameters:
    data (pd.DataFrame): Prepared features for customers
    model: Trained churn model
    features (list): List of feature names
    scaler: Fitted StandardScaler
    threshold (float): Probability threshold to consider a customer at risk

    Returns:
    pd.DataFrame: High-value customers at risk of churning
    """
    # Scale the features
    X = scaler.transform(data[features])

    # Predict churn probability
    churn_proba = model.predict_proba(X)[:, 1]
    data['churn_probability'] = churn_proba

    # Define high-value customers (e.g., top 25% by total_spend)
    high_value_threshold = data['total_spend'].quantile(0.75)
    
    # Identify high-value customers at risk of churning
    high_value_at_risk = data[(data['total_spend'] >= high_value_threshold) & 
                              (data['churn_probability'] > threshold)]

    return high_value_at_risk.sort_values('churn_probability', ascending=False)

def main() -> None:
    """
    Main function to execute the churn analysis workflow.
    """
    data = load_data()
    
    churn_labels = create_sliding_window_churn_label(data)
    
    rfm = calculate_rfm(data)
    rfm = create_churn_label(rfm)
    features = prepare_features(rfm)
    model, X_test, y_test, y_train, y_test = train_churn_model(features, rfm['Churn'])
    evaluate_model(model, X_test, y_test)
    high_value_at_risk = identify_high_value_customers(rfm, model)
    
    print("High-value customers at risk of churning:")
    print(high_value_at_risk)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='recency', y='monetary', hue='Churn_Probability', size='frequency')
    plt.title('High-Value Customers at Risk of Churning')
    plt.xlabel('Recency (days)')
    plt.ylabel('Monetary Value')
    plt.show()

if __name__ == "__main__":
    main()