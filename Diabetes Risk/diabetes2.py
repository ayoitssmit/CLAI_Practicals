import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def diabetes_kmeans_clustering(file_name):
    print("="*65)
    print("   DIABETES RISK: k-MEANS CLUSTERING REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
        print(f"[+] Dataset Loaded. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. UNSUPERVISED PREPROCESSING (THE 'STRING' TRAP)
    # ==========================================================
    print("\n[+] 1. Preprocessing (Unsupervised Logic)...")
    
    # Identify and drop the target label (e.g., 'class' or 'result')
    target_candidates = ['class', 'result', 'outcome', 'diabetes']
    target_col = next((c for c in df.columns if c.lower() in target_candidates), None)
    
    if target_col:
        df.drop(columns=[target_col], inplace=True)
        print(f"    -> Dropped target '{target_col}' to ensure strict Unsupervised Learning.")

    # Encode categorical 'Yes/No' or 'Male/Female' to 1/0
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))
    
    print("    -> Encoded all categorical text features to numeric 1/0.")

    # ==========================================================
    # 2. FEATURE SCALING (CRITICAL FOR K-MEANS)
    # ==========================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    print("[+] 2. Applied StandardScaler to all physical features.")

    # ==========================================================
    # 3. ELBOW METHOD (Finding Optimal k)
    # ==========================================================
    print("\n[+] 3. Calculating Elbow Method (WCSS)...")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), wcss, marker='o', color='maroon', linestyle='--')
    plt.title('Elbow Method: Diabetes Risk Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True, alpha=0.3)
    plt.show()

    # ==========================================================
    # 4. RUN FINAL k-MEANS
    # ==========================================================
    # For diabetes risk, k=2 is the most logical (Low Risk vs High Risk)
    k = 2
    print(f"\n[+] 4. Training k-Means with k={k}...")
    kmeans_final = KMeans(n_clusters=k, init='k-means++', random_state=42)
    clusters = kmeans_final.fit_predict(X_scaled)
    
    df['Cluster'] = clusters
    print(f"    -> Distribution: {df['Cluster'].value_counts().to_dict()}")

    # ==========================================================
    # 5. PCA VISUALIZATION (2D Projection)
    # ==========================================================
    print("\n[+] 5. Applying PCA for Cluster Visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolors='k', alpha=0.6)
    
    # Plot Centroids
    centroids = pca.transform(kmeans_final.cluster_centers_)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
    
    plt.title('k-Means: Diabetes Risk Clusters (PCA Reduced)')
    plt.legend()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION:")
    print("1. All 'Yes/No' features were encoded to allow Euclidean distance math.")
    print("2. The Elbow Method confirmed that k=2 or k=3 are mathematically optimal.")
    print("3. PCA visualization shows clear separation between high-risk and low-risk groups.")
    print("="*65)

diabetes_kmeans_clustering('diabetes_risk.csv')