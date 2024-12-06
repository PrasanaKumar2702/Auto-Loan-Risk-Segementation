
# Import required libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

# Data Preprocessing Function
def preprocess_data(input_path, output_path):
    data = pd.read_csv(input_path)
    
    # Standardize numerical columns
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Save preprocessed data
    pd.DataFrame(data_scaled, columns=data.columns).to_csv(output_path, index=False)
    print("Data preprocessing completed. Processed data saved to:", output_path)

# Clustering Function
def perform_clustering(input_path, output_dir, n_clusters=3):
    data = pd.read_csv(input_path)
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(data)
    
    # Save clustered data
    clustered_data_path = os.path.join(output_dir, "clustered_data.csv")
    data.to_csv(clustered_data_path, index=False)
    print("Clustering completed. Clustered data saved to:", clustered_data_path)
    
    # Calculate and save silhouette score
    silhouette_avg = silhouette_score(data.drop(columns=['Cluster']), data['Cluster'])
    print("Silhouette Score:", silhouette_avg)
    with open(os.path.join(output_dir, "silhouette_score.txt"), "w") as f:
        f.write(f"Silhouette Score: {silhouette_avg}")
    
    # Plot cluster centers
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(kmeans.cluster_centers_)), kmeans.inertia_, color='blue')
    plt.title("Cluster Centers")
    plt.xlabel("Cluster Index")
    plt.ylabel("Inertia")
    plt.savefig(os.path.join(output_dir, "cluster_centers.png"))
    print("Cluster center plot saved to:", output_dir)

# Main Function
if __name__ == "__main__":
    # File paths
    input_file = "data/borrower_data.csv"
    processed_file = "data/processed_borrower_data.csv"
    output_dir = "results"

    # Create directories if they do not exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Preprocess data
    preprocess_data(input_file, processed_file)

    # Step 2: Perform clustering
    perform_clustering(processed_file, output_dir)
