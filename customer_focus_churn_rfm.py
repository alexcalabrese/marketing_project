import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import yaml
import os
from typing import List, Tuple
from datetime import timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path: str = 'merged_data.csv', sample: float = None) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file, by default 'merged_data.csv'.
    sample : float, optional
        Sample size as a fraction of the dataset, by default None.

    Returns
    -------
    pd.DataFrame
        Loaded data as a DataFrame.
    """
    with open('configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    full_path = os.path.join(config['data_paths']['base_path'], path)
    data = pd.read_csv(full_path)
    
    if sample is not None:
        data = data.sample(frac=sample, random_state=42)  # Sample the dataset
    
    return data

def prepare_features_no_window(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for model training based on the sliding window churn labels.
    
    Parameters:
    data (pd.DataFrame): Original purchase data
    churn_labels (pd.DataFrame): Churn labels from sliding window approach
    
    Returns:
    pd.DataFrame: Prepared features for each customer-window combination
    """
    # Calculate features
    customer_features = data.groupby('customer_id').agg({
        'purchase_datetime': 'count',  # Frequency
        'gross_price': ['sum', 'mean', 'max']  # Monetary
    }).reset_index()
    
    customer_features.columns = ['customer_id', 'frequency', 'total_monthly_spend', 'avg_spend', 'max_spend']
    
    # Calculate recency
    last_purchase = data.groupby('customer_id')['purchase_datetime'].max().reset_index()
    last_purchase['recency'] = (data['purchase_datetime'].max() - last_purchase['purchase_datetime']).dt.days
    
    # Merge features
    customer_features = customer_features.merge(last_purchase[['customer_id', 'recency']], on='customer_id')
    
    return customer_features

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

    Parameters
    ----------
    data : pd.DataFrame
        Original purchase data.
    churn_labels : pd.DataFrame
        Churn labels from sliding window approach.

    Returns
    -------
    pd.DataFrame
        Prepared features for each customer-window combination.
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
            'gross_price': ['mean', 'max']  # Monetary
        }).reset_index()
        
        customer_features.columns = ['customer_id', 'frequency', 'avg_spend', 'max_spend']
        
        # Calculate recency
        last_purchase = window_data.groupby('customer_id')['purchase_datetime'].max().reset_index()
        last_purchase['recency'] = (window_end - last_purchase['purchase_datetime']).dt.days
        
        # Calculate total monthly spend
        window_data['month'] = window_data['purchase_datetime'].dt.to_period('M')
        monthly_spend = window_data.groupby(['customer_id', 'month'])['gross_price'].sum().reset_index()
        total_monthly_spend = monthly_spend.groupby('customer_id')['gross_price'].mean().reset_index()
        total_monthly_spend.columns = ['customer_id', 'total_monthly_spend']
        
        # Merge features
        customer_features = customer_features.merge(last_purchase[['customer_id', 'recency']], on='customer_id')
        customer_features = customer_features.merge(total_monthly_spend, on='customer_id')
        
        # Add window information
        customer_features['window_start'] = window_start
        customer_features['window_end'] = window_end
        
        features.append(customer_features)
    
    features_df = pd.concat(features, ignore_index=True)
    
    # Merge with churn labels
    final_features = features_df.merge(churn_labels, on=['customer_id', 'window_start', 'window_end'])
    
    return final_features

def preprocess_data(data: pd.DataFrame, preprocessor: ColumnTransformer = None) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Preprocess the data to include additional features.

    Parameters
    ----------
    data : pd.DataFrame
        Original purchase data.

    Returns
    -------
    pd.DataFrame
        Preprocessed data with additional features.
    ColumnTransformer
        The fitted preprocessor.
    """
    # One-hot encode categorical features
    categorical_features = ['product_class', 'job_type', 'region', 'loyalty_type']
    binary_features = ['gender']
    numerical_features = ['price_reduction', 'flag_phone_provided', 'loyatlty_status']

    # Ensure all expected columns are present
    expected_columns = categorical_features + binary_features + numerical_features + ['email_provider', 'customer_id']
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in input data: {missing_columns}")

    # Preprocess email_provider to keep top 3 and label others as 'others'
    top_3_providers = data['email_provider'].value_counts().nlargest(3).index
    data['email_provider'] = data['email_provider'].apply(lambda x: x if x in top_3_providers else 'others')

    # Preprocessing pipelines for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('binary', OneHotEncoder(drop='if_binary', sparse_output=False))
    ])

    # Combine preprocessing steps
    if preprocessor is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features + ['email_provider']),
                ('bin', binary_transformer, binary_features)
            ],
            sparse_threshold=0
        )
        data_preprocessed = preprocessor.fit_transform(data)
    else:
        data_preprocessed = preprocessor.transform(data)

    # Debug: Print the shape of the transformed data
    print(f"Shape of transformed data: {data_preprocessed.shape}")
    print(f"Expected number of columns: {len(preprocessor.get_feature_names_out())}")

    # Convert to DataFrame
    data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=preprocessor.get_feature_names_out())

    # Add the customer_id column back to the DataFrame
    data_preprocessed_df['customer_id'] = data['customer_id'].values

    return data_preprocessed_df, preprocessor

def train_churn_model(dataset: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[RandomForestClassifier, List[str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], StandardScaler]:
    """
    Train a Random Forest model to predict churn.

    Parameters
    ----------
    dataset : pd.DataFrame
        Prepared features and churn labels

    Returns
    -------
    tuple
        Trained model, feature names, and train/test split data and scaler
    """
    features = ['recency', 'frequency', 'total_monthly_spend', 'avg_spend', 'max_spend'] + list(preprocessor.get_feature_names_out())
    
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

    return model, features, (X_train_scaled, X_test_scaled, y_train, y_test), scaler

def train_logistic_regression_model(dataset: pd.DataFrame, preprocessor: ColumnTransformer) -> Tuple[LogisticRegression, List[str], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], StandardScaler]:
    """
    Train a Logistic Regression model to predict churn.

    Parameters
    ----------
    dataset : pd.DataFrame
        Prepared features and churn labels

    Returns
    -------
    tuple
        Trained model, feature names, and train/test split data and scaler
    """
    from sklearn.linear_model import LogisticRegression

    features = ['recency', 'frequency', 'total_monthly_spend', 'avg_spend', 'max_spend'] + list(preprocessor.get_feature_names_out())
    
    X = dataset[features]
    y = dataset['churned']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, features, (X_train_scaled, X_test_scaled, y_train, y_test), scaler



def evaluate_model(model: RandomForestClassifier, X_test: np.ndarray, y_test: pd.Series):
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

    return {
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'roc_auc': roc_auc
    }

def identify_high_value_customers(data: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """
    Identify high-value customers at risk of churning and cluster them based on RFM values.

    Parameters:
    data (pd.DataFrame): Prepared features for customers
    model: Trained churn model
    features (list): List of feature names
    scaler: Fitted StandardScaler
    threshold (float): Probability threshold to consider a customer at risk

    Returns:
    pd.DataFrame: High-value customers at risk of churning with RFM clusters
    """
    # Scale the features
    # X = scaler.transform(data[features])

    # # Predict churn probability
    # churn_proba = model.predict_proba(X)[:, 1]
    # data['churn_probability'] = churn_proba

    # Define high-value customers (e.g., top 25% by total_monthly_spend)
    high_value_threshold = data['total_monthly_spend'].quantile(0.75)
    
    # Identify high-value customers at risk of churning
    high_value_at_risk = data[(data['total_monthly_spend'] >= high_value_threshold) & 
                               (data['churn_probability_rf'] > threshold)]

    # Scale RFM values between 0 and 1
    rfm_scaled = (high_value_at_risk[['recency', 'total_monthly_spend', 'frequency']] - high_value_at_risk[['recency', 'total_monthly_spend', 'frequency']].min()) / (high_value_at_risk[['recency', 'total_monthly_spend', 'frequency']].max() - high_value_at_risk[['recency', 'total_monthly_spend', 'frequency']].min())

    # Calculate weighted average of scaled RFM values
    rfm_scaled['RFM_avg'] = rfm_scaled.mean(axis=1)

    # Cluster customers into 3 groups based on weighted average RFM values
    high_value_at_risk['RFM_cluster'] = pd.qcut(rfm_scaled['RFM_avg'], q=3, labels=["Low", "Medium", "High"])

    return high_value_at_risk.sort_values('churn_probability_rf', ascending=False)

def main() -> None:
    """
    Main function to execute the churn analysis workflow.
    """
    
    print("Loading data...")
    data = load_data(sample=0.01)
    rfm = calculate_rfm(data)

    print("Creating sliding window churn labels...")
    # Create sliding window churn labels
    churn_labels = create_sliding_window_churn_label(data)

    print("Preparing features...")
    # Prepare features
    features_df = prepare_features(data, churn_labels)
    
    print("Preprocessing additional features...")
    # Preprocess additional features
    additional_features_df, preprocessor = preprocess_data(data)
    features_df = features_df.merge(additional_features_df, on='customer_id', how='left')
    
    print("Training model...")
    # Train the model
    model, feature_names, (X_train, X_test, y_train, y_test), scaler = train_churn_model(features_df, preprocessor)

    # filter the data for the last 90 days
    last_window_data = data[(data['purchase_datetime'] >= data['purchase_datetime'].max() - timedelta(days=90))]
    
    # Prepare features for the current data
    last_window_features = prepare_features_no_window(last_window_data)
    
    # Preprocess the last window features
    last_window_features_preprocessed, _ = preprocess_data(last_window_data, preprocessor)
    
    last_window_features_df = last_window_features_preprocessed.merge(last_window_features, on='customer_id', how='left')
    
    # Predict churn probabilities using the trained model
    churn_proba_last_window = model.predict_proba(last_window_features_df[feature_names])[:, 1]
    
    # Add churn probabilities to the last window features DataFrame
    last_window_features_df['churn_probability'] = churn_proba_last_window
    
    # Print the top 10 customers with the highest churn probability
    top_churn_risk_customers = last_window_features_df.sort_values('churn_probability', ascending=False).head(10)
    print("Top 10 customers with the highest churn probability:")
    print(top_churn_risk_customers)


if __name__ == "__main__":
    main()