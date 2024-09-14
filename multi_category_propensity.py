import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from typing import Tuple, List, Dict
from datetime import timedelta
import yaml
import os
import logging
from joblib import dump

logging.basicConfig(level=logging.INFO)

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
    with open('/teamspace/studios/this_studio/marketing_repo/marketing_project/configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    full_path = os.path.join(config['data_paths']['base_path'], path)
    data = pd.read_csv(full_path)
    
    if sample is not None:
        data = data.sample(frac=sample, random_state=42)
    
    return data

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, ColumnTransformer]:
    """
    Preprocess the data for propensity modeling.

    Parameters
    ----------
    data : pd.DataFrame
        Original purchase data.

    Returns
    -------
    Tuple[pd.DataFrame, ColumnTransformer]
        Preprocessed data and the fitted preprocessor.
    """
    categorical_features = ['job_type', 'region', 'loyalty_type']
    binary_features = ['gender']
    numerical_features = ['price_reduction', 'flag_phone_provided', 'loyatlty_status']

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

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features)
        ],
        sparse_threshold=0
    )

    data_preprocessed = preprocessor.fit_transform(data)
    data_preprocessed_df = pd.DataFrame(data_preprocessed, columns=preprocessor.get_feature_names_out())
    
    data_preprocessed_df['customer_id'] = data['customer_id'].values
    data_preprocessed_df['purchase_datetime'] = data['purchase_datetime'].values
    data_preprocessed_df['product_class'] = data['product_class'].values

    return data_preprocessed_df, preprocessor

def create_multi_category_target(data: pd.DataFrame, observation_window: int = 120, 
                                 prediction_window: int = 60, slide_step: int = 30) -> pd.DataFrame:
    """
    Create multi-category target variable for propensity to purchase model using a sliding window approach.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed purchase data.
    observation_window : int, optional
        Number of days in the observation period, by default 120.
    prediction_window : int, optional
        Number of days in the prediction period, by default 60.
    slide_step : int, optional
        Number of days to slide the window, by default 30.

    Returns
    -------
    pd.DataFrame
        Data with multi-category target variable added for multiple periods.
    """
    data['purchase_datetime'] = pd.to_datetime(data['purchase_datetime'])
    start_date = data['purchase_datetime'].min()
    end_date = data['purchase_datetime'].max()

    all_periods = []

    while start_date + timedelta(days=observation_window + prediction_window) <= end_date:
        period_end = start_date + timedelta(days=observation_window)
        prediction_end = period_end + timedelta(days=prediction_window)

        observation_data = data[(data['purchase_datetime'] >= start_date) & (data['purchase_datetime'] < period_end)]
        prediction_data = data[(data['purchase_datetime'] >= period_end) & (data['purchase_datetime'] < prediction_end)]

        target = prediction_data.groupby('customer_id')['product_class'].apply(set).reset_index()
        target.columns = ['customer_id', 'target_categories']

        result = observation_data.merge(target, on='customer_id', how='left')
        result['target_categories'] = result['target_categories'].fillna('').apply(list)
        result['period_start'] = start_date
        result['period_end'] = period_end

        all_periods.append(result)

        start_date += timedelta(days=slide_step)

    final_result = pd.concat(all_periods, ignore_index=True)
    final_result = final_result.drop(['purchase_datetime'], axis=1)

    return final_result

def train_multi_category_propensity_model(data: pd.DataFrame, features: List[str]) -> Tuple[MultiOutputClassifier, StandardScaler, MultiLabelBinarizer]:
    """
    Train a Multi-label Random Forest model to predict propensity to purchase for multiple categories.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared features and target variable.
    features : List[str]
        List of feature names to use in the model.

    Returns
    -------
    Tuple[MultiOutputClassifier, StandardScaler, MultiLabelBinarizer]
        Trained model, fitted scaler, and fitted MultiLabelBinarizer.
    """
    X = data[features]
    y = data['target_categories']

    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', verbose=1)
    model = MultiOutputClassifier(base_model)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(num) for num in mlb.classes_.tolist()]))

    print("\nConfusion Matrix:")
    for i, category in enumerate(mlb.classes_):
        print(f"\nCategory: {category}")
        print(confusion_matrix(y_test[:, i], y_pred[:, i]))

    # Calculate ROC AUC for each category
    for i, category in enumerate(mlb.classes_):
        roc_auc = roc_auc_score(y_test[:, i], y_pred_proba[i][:, 1])
        print(f"\nROC AUC Score for {category}: {roc_auc:.4f}")

    return model, scaler, mlb

def predict_category_propensities(customer_data: pd.DataFrame, model: MultiOutputClassifier, 
                                  scaler: StandardScaler, mlb: MultiLabelBinarizer, 
                                  features: List[str]) -> pd.DataFrame:
    """
    Predict propensities for each category for given customers.

    Parameters
    ----------
    customer_data : pd.DataFrame
        Customer data for prediction.
    model : MultiOutputClassifier
        Trained multi-label classification model.
    scaler : StandardScaler
        Fitted scaler for feature scaling.
    mlb : MultiLabelBinarizer
        Fitted MultiLabelBinarizer for decoding predictions.
    features : List[str]
        List of feature names used in the model.

    Returns
    -------
    pd.DataFrame
        DataFrame with predicted propensities for each category for each customer.
    """
    X = customer_data[features]
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)

    results = []

    for i, category in enumerate(mlb.classes_):
        category_probs = probabilities[i][:, 1]
        
        result = pd.DataFrame({
            'customer_id': customer_data['customer_id'],
            f'propensity_{category}': category_probs
        })
        results.append(result)

    all_results = pd.concat(results, axis=1)
    all_results = all_results.loc[:,~all_results.columns.duplicated()]  # Remove duplicate customer_id columns
    
    return all_results

def main():
    # Load and preprocess data
    sample = 1
    data = load_data(sample=sample)
    preprocessed_data, preprocessor = preprocess_data(data)

    # Create multi-category target variable using sliding window
    multi_category_target_data = create_multi_category_target(preprocessed_data,
                                                              observation_window=120, 
                                                              prediction_window=60, 
                                                              slide_step=30)

    # Define features
    features = preprocessor.get_feature_names_out().tolist()

    # Log start of training process
    logging.info("Starting multi-category model training process")

    # Train multi-category model
    model, scaler, mlb = train_multi_category_propensity_model(multi_category_target_data, features)

    # Log end of training process
    logging.info("Multi-category model training process completed successfully")

    # Predict propensities for all customers
    all_customer_propensities = predict_category_propensities(preprocessed_data, model, scaler, mlb, features)

    print("\nPredicted Propensities (first 5 customers):")
    print(all_customer_propensities.head(5))

    # Print some statistics about the predicted propensities
    print("\nPropensity Statistics:")
    for category in mlb.classes_:
        print(f"\nCategory {category}:")
        stats = all_customer_propensities[f'propensity_{category}'].describe()
        print(f"Count: {stats['count']}, Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}, "
              f"Min: {stats['min']:.4f}, 25%: {stats['25%']:.4f}, 50%: {stats['50%']:.4f}, "
              f"75%: {stats['75%']:.4f}, Max: {stats['max']:.4f}")

    # Additional analysis
    print("\nMulti-category Target distribution across all periods:")
    category_counts = multi_category_target_data['target_categories'].apply(pd.Series).stack().value_counts()
    print(category_counts / len(multi_category_target_data))

    print("\nNumber of unique periods:", multi_category_target_data['period_start'].nunique())
    
    # Save predictions to CSV
    with open('/teamspace/studios/this_studio/marketing_repo/marketing_project/configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    base_path = config['data_paths']['base_path']
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    dataset_percentage = 1 if sample is None else sample * 100
    output_file = os.path.join(base_path, f'multi_category_propensities_{timestamp}_{dataset_percentage:.2f}percent.csv')
    
    all_customer_propensities.to_csv(output_file, index=False)
    logging.info(f"Predictions saved to {output_file}")
    logging.info(f"Dataset used: {dataset_percentage:.2f}% of full dataset")
    
    # Save the model
    model_output_file = os.path.join(base_path, f'multi_category_model_{timestamp}.joblib')
    dump((model, scaler, mlb), model_output_file)
    logging.info(f"Model saved to {model_output_file}")
    

if __name__ == "__main__":
    main()