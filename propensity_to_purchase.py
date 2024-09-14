import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import yaml
import os
import logging

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
        data = data.sample(frac=sample, random_state=42)  # Sample the dataset
    
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
    categorical_features = ['product_class', 'job_type', 'region', 'loyalty_type']
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
    
    # Add back the necessary columns
    data_preprocessed_df['customer_id'] = data['customer_id'].values
    data_preprocessed_df['purchase_datetime'] = data['purchase_datetime'].values
    data_preprocessed_df['product_id'] = data['product_id'].values
    data_preprocessed_df['product_class'] = data['product_class'].values

    return data_preprocessed_df, preprocessor

def create_target_variable(data: pd.DataFrame, product: str = None, category: str = None, 
                           observation_window: int = 120, prediction_window: int = 60,
                           slide_step: int = 30) -> pd.DataFrame:
    """
    Create target variable for propensity to purchase model using a sliding window approach.

    Parameters
    ----------
    data : pd.DataFrame
        Preprocessed purchase data.
    product : str, optional
        Specific product to model propensity for, by default None.
    category : str, optional
        Product category to model propensity for, by default None.
    observation_window : int, optional
        Number of days in the observation period, by default 90.
    prediction_window : int, optional
        Number of days in the prediction period, by default 30.
    slide_step : int, optional
        Number of days to slide the window, by default 30.

    Returns
    -------
    pd.DataFrame
        Data with target variable added for multiple periods.
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

        if product:
            target = (prediction_data['product_id'] == int(product)).groupby(prediction_data['customer_id']).any()
        elif category:
            target = (prediction_data['product_class'] == int(category)).groupby(prediction_data['customer_id']).any()
        else:
            raise ValueError("Either 'product' or 'category' must be specified.")

        target = target.reset_index()
        target.columns = ['customer_id', 'target']

        result = observation_data.merge(target, on='customer_id', how='left')
        result['target'] = result['target'].fillna(0)
        result['period_start'] = start_date
        result['period_end'] = period_end

        all_periods.append(result)

        start_date += timedelta(days=slide_step)

    final_result = pd.concat(all_periods, ignore_index=True)

    # Remove the columns we added back for target creation, except for period information
    final_result = final_result.drop(['purchase_datetime', 'product_id', 'product_class'], axis=1)

    return final_result

def train_propensity_model(data: pd.DataFrame, features: List[str]) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a Random Forest model to predict propensity to purchase.

    Parameters
    ----------
    data : pd.DataFrame
        Prepared features and target variable.
    features : List[str]
        List of feature names to use in the model.

    Returns
    -------
    Tuple[RandomForestClassifier, StandardScaler]
        Trained model and fitted scaler.
    """
    X = data[features]
    y = data['target']

    # Add this line to print the total entries
    print(f"Total entries in dataset: {len(X)}")

    # Check class distribution
    class_distribution = y.value_counts(normalize=True)
    print("Class distribution:")
    print(class_distribution)

    if len(class_distribution) == 1:
        print("Warning: Only one class present in the target variable.")
        return None, None

    # Ensure target variable has only 0 or 1 values
    y = y.astype(int)
    
    # Double-check unique values after conversion
    unique_values = y.unique()
    print("Unique values in target variable after conversion:", unique_values)
    
    if not set(unique_values).issubset({0, 1}):
        raise ValueError("Target variable contains values other than 0 and 1 after conversion.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    if y_pred_proba.shape[1] > 1:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        print(f"\nROC AUC Score: {roc_auc:.4f}")
    else:
        print("\nWarning: ROC AUC Score cannot be calculated due to single class prediction.")

    return model, scaler

def calculate_vpo(probability: float, npv: float, cost: float) -> float:
    """
    Calculate the Value of Propensity Offer (VPO).

    Parameters
    ----------
    probability : float
        Probability of purchase.
    npv : float
        Net Present Value of the product.
    cost : float
        Cost of the offer.

    Returns
    -------
    float
        Calculated VPO.
    """
    # Check if probability is a scalar or an array
    if isinstance(probability, (float, int)):
        if not (0 <= probability <= 1):
            print(f"Warning: Invalid probability value: {probability}")
            return float('nan')
    else:
        # For array-like probabilities
        invalid_probs = (probability < 0) | (probability > 1)
        if invalid_probs.any():
            print(f"Warning: Invalid probability values found: {probability[invalid_probs]}")
            probability = np.where(invalid_probs, np.nan, probability)

    if np.any(npv < 0):
        print(f"Warning: Negative NPV value(s) found: {npv[npv < 0]}")
    
    if np.any(cost < 0):
        print(f"Warning: Negative cost value(s) found: {cost[cost < 0]}")
    
    return probability * npv - cost

def next_best_offer(customer_data: pd.DataFrame, models: Dict[str, RandomForestClassifier], 
                    scalers: Dict[str, StandardScaler], features: List[str], 
                    npvs: Dict[str, float], costs: Dict[str, float]) -> pd.DataFrame:
    """
    Determine the Next Best Offer for each customer.

    Parameters
    ----------
    customer_data : pd.DataFrame
        Customer data for prediction.
    models : Dict[str, RandomForestClassifier]
        Dictionary of trained models for each product/category.
    scalers : Dict[str, StandardScaler]
        Dictionary of fitted scalers for each product/category.
    features : List[str]
        List of feature names used in the models.
    npvs : Dict[str, float]
        Dictionary of Net Present Values for each product/category.
    costs : Dict[str, float]
        Dictionary of costs for each product/category offer.

    Returns
    -------
    pd.DataFrame
        DataFrame with Next Best Offer for each customer.
    """
    results = []

    for product, model in models.items():
        X = customer_data[features]
        X_scaled = scalers[product].transform(X)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        vpos = calculate_vpo(probabilities, npvs[product], costs[product])
        
        result = pd.DataFrame({
            'customer_id': customer_data['customer_id'],
            'product': product,
            'probability': probabilities,
            'vpo': vpos
        })
        results.append(result)

    all_results = pd.concat(results)
    
    # Debug: Print some information about the VPO values
    print("VPO statistics:")
    print(all_results['vpo'].describe())
    print("Number of NaN VPO values:", all_results['vpo'].isna().sum())
    
    # Handle NaN values in VPO
    all_results['vpo'] = all_results['vpo'].fillna(-float('inf'))
    
    best_offers = all_results.loc[all_results.groupby('customer_id')['vpo'].idxmax()]
    
    # Remove rows where VPO is still -inf (i.e., all offers for that customer were NaN)
    best_offers = best_offers[best_offers['vpo'] != -float('inf')]
    
    return best_offers

def main():
    # Load and preprocess data
    data = load_data(sample=1)
    preprocessed_data, preprocessor = preprocess_data(data)

    CATEGORY = 9
    PRODUCT = 48500403

    # Create target variables for both category and specific product using sliding window
    category_target_data = create_target_variable(preprocessed_data, category=CATEGORY,
                                                  observation_window=120, prediction_window=60, 
                                                  slide_step=30)
    product_target_data = create_target_variable(preprocessed_data, product=PRODUCT,
                                                 observation_window=120, prediction_window=60, 
                                                 slide_step=30)

    # Define features
    features = preprocessor.get_feature_names_out().tolist()

    # Log start of training process
    logging.info("Starting model training process")

    # Train models
    category_model, category_scaler = train_propensity_model(category_target_data, features)
    product_model, product_scaler = train_propensity_model(product_target_data, features)

    if category_model is None or product_model is None:
        logging.error("Unable to train models due to insufficient class diversity.")
        return

    # Log end of training process
    logging.info("Model training process completed successfully")

    # Calculate NPV and cost for the specific category and product
    category_data = data[data['product_class'] == CATEGORY]
    product_data = data[data['product_id'] == PRODUCT]

    def calculate_npv_and_cost(data):
        average_revenue = data['gross_price'].mean()
        profit_margin = 0.3  # Adjust based on your business knowledge
        npv = average_revenue * profit_margin

        email_cost = 0.05  # Adjust based on your actual marketing costs
        average_discount = data['price_reduction'].mean()
        fulfillment_cost = 1.00  # Adjust based on your actual fulfillment costs
        cost = email_cost + average_discount + fulfillment_cost

        return npv, cost

    category_npv, category_cost = calculate_npv_and_cost(category_data)
    product_npv, product_cost = calculate_npv_and_cost(product_data)

    print(f"Calculated Category NPV: {category_npv}")
    print(f"Calculated Category cost: {category_cost}")
    print(f"Calculated Product NPV: {product_npv}")
    print(f"Calculated Product cost: {product_cost}")

    # Example of Next Best Offer
    models = {
        'specific_category': category_model,
        'specific_product': product_model
    }
    scalers = {
        'specific_category': category_scaler,
        'specific_product': product_scaler
    }
    npvs = {
        'specific_category': category_npv,
        'specific_product': product_npv
    }
    costs = {
        'specific_category': category_cost,
        'specific_product': product_cost
    }

    # Get Next Best Offer for each customer
    next_best_offers = next_best_offer(preprocessed_data, models, scalers, features, npvs, costs)
    print("Next Best Offers (top 10 probabilities):")
    print(next_best_offers.sort_values('probability', ascending=False).head(10))

    # Print some statistics about the next best offers
    print("\nNext Best Offers statistics:")
    print(next_best_offers['probability'].describe())
    print(next_best_offers['vpo'].describe())

    # Additional analysis
    print("\nCategory Target distribution across all periods:")
    print(category_target_data['target'].value_counts(normalize=True))
    print("\nProduct Target distribution across all periods:")
    print(product_target_data['target'].value_counts(normalize=True))

    print("\nNumber of unique periods (Category):", category_target_data['period_start'].nunique())
    print("Number of unique periods (Product):", product_target_data['period_start'].nunique())

    # Print calculated NPV and cost for the specific category and product
    print(f"\nCalculated NPV for category {CATEGORY}: {category_npv:.2f}")
    print(f"Calculated cost for category {CATEGORY}: {category_cost:.2f}")
    print(f"Calculated NPV for product {PRODUCT}: {product_npv:.2f}")
    print(f"Calculated cost for product {PRODUCT}: {product_cost:.2f}")

if __name__ == "__main__":
    main()