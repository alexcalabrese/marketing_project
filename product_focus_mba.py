import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import yaml
import os

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

def prepare_transaction_data(data):
    # Group the data by order_id and aggregate product_ids into a list
    transactions = data.groupby('order_id')['product_id'].agg(lambda x: list(set(x))).reset_index()
    
    # Use MultiLabelBinarizer for one-hot encoding
    mlb = MultiLabelBinarizer(sparse_output=True)
    transaction_matrix = mlb.fit_transform(transactions['product_id'])
    
    # Create a DataFrame with the binary encoded data
    transactions_encoded = pd.DataFrame.sparse.from_spmatrix(
        transaction_matrix,
        columns=mlb.classes_,
        index=transactions['order_id']
    )
    
    return transactions_encoded

def perform_mba(transactions, min_support=0.01, min_threshold=0.7):
    # Cast the column names as strings
    transactions.columns = [str(col) for col in transactions.columns]
    
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    
    print(f"Frequent itemsets found: {len(frequent_itemsets)}")

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules

def identify_cross_selling_opportunities(rules, top_n=10):
    cross_sell_opportunities = rules.sort_values('lift', ascending=False).head(top_n)
    return cross_sell_opportunities

def visualize_network(rules, min_lift=2, max_nodes=20):
    import networkx as nx
    
    # Filter rules and limit the number of nodes
    filtered_rules = rules[rules['lift'] >= min_lift].head(max_nodes)
    
    G = nx.Graph()
    for i, row in filtered_rules.iterrows():
        G.add_edge(','.join(row['antecedents']), ','.join(row['consequents']), weight=row['lift'])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, font_weight='bold')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.title('Product Association Network')
    plt.axis('off')
    # plt.tight_layout()
    
    plt.show()

def main():
    print("Loading data...")
    data = load_data()
    
    print("Preparing transaction data...")
    transactions = prepare_transaction_data(data)
    
    print("Performing Market Basket Analysis...")
    rules = perform_mba(transactions, min_support=0.003, min_threshold=0.3)
    
    print("Identifying cross-selling opportunities...")
    cross_sell_opportunities = identify_cross_selling_opportunities(rules)
    
    print("\nTop cross-selling opportunities:")
    print(cross_sell_opportunities[['antecedents', 'consequents', 'lift', 'confidence']])
    
    # print("\nVisualizing network...")
    # visualize_network(rules)

if __name__ == "__main__":
    main()