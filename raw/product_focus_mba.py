import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def load_data():
    return pd.read_csv('merged_data.csv')

def prepare_transaction_data(data):
    transactions = data.groupby(['order_id', 'product_id'])['product_class'].first().unstack().reset_index()
    transactions = transactions.fillna(0).set_index('order_id')
    return transactions

def perform_mba(transactions, min_support=0.01, min_threshold=0.7):
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    return rules

def identify_cross_selling_opportunities(rules, top_n=10):
    cross_sell_opportunities = rules.sort_values('lift', ascending=False).head(top_n)
    return cross_sell_opportunities

def visualize_network(rules, min_lift=2):
    import networkx as nx
    
    G = nx.Graph()
    for i, row in rules[rules['lift'] >= min_lift].iterrows():
        G.add_edge(','.join(row['antecedents']), ','.join(row['consequents']), weight=row['lift'])
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=3000, font_size=8, font_weight='bold')
    edge_weights = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
    plt.title('Product Association Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    data = load_data()
    transactions = prepare_transaction_data(data)
    rules = perform_mba(transactions)
    cross_sell_opportunities = identify_cross_selling_opportunities(rules)
    
    print("Top cross-selling opportunities:")
    print(cross_sell_opportunities[['antecedents', 'consequents', 'lift', 'confidence']])
    
    visualize_network(rules)

if __name__ == "__main__":
    main()