import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import logging
from joblib import load
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)

def load_propensity_data(path: str) -> pd.DataFrame:
    """
    Load propensity data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file containing propensity data.

    Returns
    -------
    pd.DataFrame
        Loaded propensity data.
    """
    return pd.read_csv(path)

def identify_primary_secondary_categories(data: pd.DataFrame, secondary_threshold: float = 0.8) -> pd.DataFrame:
    """
    Identify primary and secondary categories for each user.

    Parameters
    ----------
    data : pd.DataFrame
        Propensity data for users across categories.
    secondary_threshold : float, optional
        Threshold for identifying secondary categories, by default 0.8

    Returns
    -------
    pd.DataFrame
        DataFrame with primary and secondary categories for each user.
    """
    category_columns = [col for col in data.columns if col.startswith('propensity_')]
    
    results = []
    for _, row in data.iterrows():
        propensities = row[category_columns].sort_values(ascending=False)
        primary_category = propensities.index[0].split('_')[1]
        secondary_categories = propensities[propensities >= secondary_threshold * propensities.iloc[0]].index[1:].tolist()
        secondary_categories = [cat.split('_')[1] for cat in secondary_categories]
        
        results.append({
            'customer_id': row['customer_id'],
            'primary_category': primary_category,
            'secondary_categories': secondary_categories
        })
    
    return pd.DataFrame(results)

def perform_cluster_segmentation(data: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    """
    Perform cluster segmentation on users based on their category propensities.

    Parameters
    ----------
    data : pd.DataFrame
        Propensity data for users across categories.
    n_clusters : int, optional
        Number of clusters to form, by default 5

    Returns
    -------
    Tuple[pd.DataFrame, KMeans]
        DataFrame with cluster assignments and the fitted KMeans model.
    """
    category_columns = [col for col in data.columns if col.startswith('propensity_')]
    X = data[category_columns]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    data['cluster'] = cluster_labels
    return data, kmeans

def analyze_clusters(data: pd.DataFrame, kmeans: KMeans):
    """
    Analyze and print insights about the formed clusters.

    Parameters
    ----------
    data : pd.DataFrame
        Propensity data with cluster assignments.
    kmeans : KMeans
        Fitted KMeans model.
    """
    category_columns = [col for col in data.columns if col.startswith('propensity_')]
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=category_columns)
    
    for cluster in range(len(cluster_centers)):
        print(f"\nCluster {cluster}:")
        top_categories = cluster_centers.iloc[cluster].nlargest(3)
        print("Top 3 categories:")
        for cat, score in top_categories.items():
            print(f"  {cat.split('_')[1]}: {score:.4f}")
        
        cluster_size = (data['cluster'] == cluster).sum()
        print(f"Cluster size: {cluster_size} ({cluster_size/len(data)*100:.2f}% of total)")

def visualize_clusters_3d(data: pd.DataFrame, kmeans: KMeans, base_path: str = None):
    """
    Visualize clusters using PCA for dimensionality reduction in 3D.

    Parameters
    ----------
    data : pd.DataFrame
        Propensity data with cluster assignments.
    kmeans : KMeans
        Fitted KMeans model.
    """
    from sklearn.decomposition import PCA
    
    category_columns = [col for col in data.columns if col.startswith('propensity_')]
    X = data[category_columns]
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                         c=data['cluster'], cmap='viridis', 
                         s=50, alpha=0.6)
    
    ax.set_title('3D Cluster Visualization (PCA)', fontsize=16)
    ax.set_xlabel('First Principal Component', fontsize=12)
    ax.set_ylabel('Second Principal Component', fontsize=12)
    ax.set_zlabel('Third Principal Component', fontsize=12)
    
    # Add a color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Cluster', fontsize=12)
    
    # Improve the angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    # Save the plot
    plot_path = os.path.join(base_path, f'cluster_visualization_3d_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"3D cluster visualization saved to {plot_path}")

def partition_data(data: pd.DataFrame, partition_size: int) -> pd.DataFrame:
    """
    Partition the data for testing with a low number of rows.

    Parameters
    ----------
    data : pd.DataFrame
        Full propensity data.
    partition_size : int
        Number of rows to include in the partition.

    Returns
    -------
    pd.DataFrame
        Partitioned data.
    """
    return data.sample(n=min(partition_size, len(data)), random_state=42)

def main():
    # Load configuration
    with open('/teamspace/studios/this_studio/marketing_repo/marketing_project/configs.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    base_path = config['data_paths']['base_path']
    
    # Load the most recent propensity data file
    propensity_files = [f for f in os.listdir(base_path) if f.startswith('multi_category_propensities_')]
    latest_file = max(propensity_files, key=lambda x: os.path.getctime(os.path.join(base_path, x)))
    propensity_data_path = os.path.join(base_path, latest_file)
    
    logging.info(f"Loading propensity data from {propensity_data_path}")
    propensity_data = load_propensity_data(propensity_data_path)
    
    # Partition data for testing (adjust partition_size as needed)
    partition_size = 99999999#1000  # Adjust this value based on your needs
    partitioned_data = partition_data(propensity_data, partition_size)
    logging.info(f"Partitioned data size: {len(partitioned_data)} rows")
    
    # Identify primary and secondary categories
    category_analysis = identify_primary_secondary_categories(partitioned_data)
    print("\nPrimary and Secondary Categories (first 5 rows):")
    print(category_analysis.head())
    
    # Perform cluster segmentation
    clustered_data, kmeans_model = perform_cluster_segmentation(partitioned_data)
    
    # Analyze clusters
    print("\nCluster Analysis:")
    analyze_clusters(clustered_data, kmeans_model)
    
    # Visualize clusters in 3D
    visualize_clusters_3d(clustered_data, kmeans_model, base_path)
    
    # Save results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    category_analysis_file = os.path.join(base_path, f'category_analysis_{timestamp}.csv')
    category_analysis.to_csv(category_analysis_file, index=False)
    logging.info(f"Category analysis saved to {category_analysis_file}")
    
    clustered_data_file = os.path.join(base_path, f'clustered_data_{timestamp}.csv')
    clustered_data.to_csv(clustered_data_file, index=False)
    logging.info(f"Clustered data saved to {clustered_data_file}")

if __name__ == "__main__":
    main()