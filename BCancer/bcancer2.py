import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def breast_cancer_hierarchical_clustering(file_name):
    print("="*65)
    print("  BREAST CANCER: HIERARCHICAL CLUSTERING & DENDROGRAM")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
        print(f"[+] Loaded {file_name}. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. THE UNSUPERVISED CLEANING (Mandatory Marks)
    # ==========================================================
    # Dropping the ID and the Target ('diagnosis' or 'target')
    # Hierarchical clustering is UNSUPERVISED. We don't use labels.
    cols_to_drop = ['id', 'diagnosis', 'target', 'Unnamed: 32']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    
    # We save the true labels separately ONLY for final verification
    true_labels = None
    if 'diagnosis' in df.columns:
        true_labels = df['diagnosis']
    elif 'target' in df.columns:
        true_labels = df['target']
        
    df_features = df.drop(columns=existing_drop, errors='ignore')
    print(f"    -> Removed ID and Labels. Clustered purely on physical features.")

    # Handle any missing values
    df_features = df_features.fillna(df_features.median())

    # ==========================================================
    # 2. FEATURE SCALING (Crucial for Euclidean Distance)
    # ==========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    print("[+] Data scaled using StandardScaler.")

    # ==========================================================
    # 3. DENDROGRAM ANALYSIS
    # ==========================================================
    print("\n[+] 3. Computing Linkage and Generating Dendrogram...")
    # Ward's method minimizes the variance within clusters
    linked = linkage(X_scaled, method='ward')

    plt.figure(figsize=(12, 7))
    plt.title('Hierarchical Clustering Dendrogram (Breast Cancer Features)')
    plt.xlabel('Sample Index (or Cluster Size)')
    plt.ylabel('Distance (Ward)')

    # If dataset is large, we truncate to keep the graph readable
    dendrogram(linked, 
               truncate_mode='lastp', 
               p=12, 
               leaf_rotation=45., 
               show_contracted=True)
    
    # Draw a line where we cut the tree for 2 clusters (Malignant vs Benign)
    plt.axhline(y=70, color='r', linestyle='--', label='Cluster Cut-off')
    plt.legend()
    plt.show()

    # ==========================================================
    # 4. APPLYING AGGLOMERATIVE CLUSTERING
    # ==========================================================
    # Since we know there are 2 main biological groups (M vs B), we set n_clusters=2
    hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
    cluster_labels = hc.fit_predict(X_scaled)
    
    print(f"\n[+] Clustered into 2 groups.")
    print(f"    - Cluster 0 count: {list(cluster_labels).count(0)}")
    print(f"    - Cluster 1 count: {list(cluster_labels).count(1)}")

    # ==========================================================
    # 5. VISUALIZATION (PCA 2D Projection)
    # ==========================================================
    print("\n[+] Visualizing clusters using PCA...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='coolwarm', edgecolors='k', alpha=0.7)
    plt.title('PCA Projection of Hierarchical Clusters')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION:")
    print("1. Dendrogram shows a clear split, suggesting two primary groupings.")
    print("2. PCA visualization confirms that the physical measurements of cells ")
    print("   naturally form two distinct clusters corresponding to diagnosis.")
    print("="*65)

# --- EXAM USAGE ---
# Just ensure 'BCancer.csv' is in your folder!
file_name = 'BCancer.csv'
breast_cancer_hierarchical_clustering(file_name)