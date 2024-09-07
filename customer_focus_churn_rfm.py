import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_data(path='merged_data.csv'):
    
    with open('/teamspace/studios/this_studio/marketing_project/configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    full_path = os.path.join(config['data_paths']['base_path'], path)
    return pd.read_csv(full_path)

def calculate_rfm(data):
    current_date = pd.Timestamp('2023-06-01')  # Adjust this to your reference date
    rfm = data.groupby('customer_id').agg({
        'purchase_datetime': lambda x: (current_date - pd.to_datetime(x.max())).days,
        'order_id': 'count',
        'gross_price': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def create_churn_label(rfm, churn_threshold=90):
    rfm['Churn'] = rfm['Recency'] > churn_threshold
    return rfm

def prepare_features(rfm):
    features = rfm[['Recency', 'Frequency', 'Monetary']]
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled

def train_churn_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def identify_high_value_customers(rfm, model, threshold=0.7):
    features = prepare_features(rfm)
    churn_proba = model.predict_proba(features)[:, 1]
    rfm['Churn_Probability'] = churn_proba
    high_value_at_risk = rfm[(rfm['Monetary'] > rfm['Monetary'].quantile(0.75)) & 
                             (rfm['Churn_Probability'] > threshold)]
    return high_value_at_risk

def main():
    data = load_data()
    rfm = calculate_rfm(data)
    rfm = create_churn_label(rfm)
    features = prepare_features(rfm)
    model, X_test, y_test = train_churn_model(features, rfm['Churn'])
    evaluate_model(model, X_test, y_test)
    high_value_at_risk = identify_high_value_customers(rfm, model)
    
    print("High-value customers at risk of churning:")
    print(high_value_at_risk)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='Churn_Probability', size='Frequency')
    plt.title('High-Value Customers at Risk of Churning')
    plt.xlabel('Recency (days)')
    plt.ylabel('Monetary Value')
    plt.show()

if __name__ == "__main__":
    main()