import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def employee_kmeans_clustering(file_name):
    print("="*65)
    print("    EMPLOYEE SALARY: k-MEANS CLUSTERING REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. PREPROCESSING (Unsupervised - NO TARGET VARIABLE!)
    # ==========================================================
    print("\n[+] 1. Preprocessing Data for Unsupervised Learning...")
    
    # Drop Employee IDs or Names (They ruin distance calculations)
    ids = ['id', 'emp_id', 'employee_id', 'name', 'first_name', 'last_name']
    df.drop(columns=[c for c in df.columns if c.lower() in ids], inplace=True, errors='ignore')

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # One-Hot Encode Categorical Variables (Department, Gender, etc.)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df = df.astype(float) # Ensure all data is numeric
        print(f"    -> Applied One-Hot Encoding to categorical features.")

    # ==========================================================
    # 2. FEATURE SCALING (CRITICAL FOR K-MEANS)
    # ==========================================================
    print("[+] 2. Scaling Features (Mandatory for Distance-Based Algorithms)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ==========================================================
    # 3. FINDING OPTIMAL 'k' USING THE ELBOW METHOD
    # ==========================================================
    print("\n[+] 3. Calculating the Elbow Method to find optimal 'k'...")
    wcss = [] # Within-Cluster Sum of Squares
    
    # Test k from 1 to 10
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        
    # Plotting the Elbow Graph
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
    plt.title('The Elbow Method (Finding optimal clusters)')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS (Inertia)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # ==========================================================
    # 4. APPLYING K-MEANS WITH OPTIMAL 'k'
    # ==========================================================
    # Based on standard employee datasets, k=3 is usually the most logical 
    # (e.g., Junior, Mid-Level, Senior employee clusters)
    optimal_k = 3 
    print(f"\n[+] 4. Training final k-Means model with k={optimal_k}...")
    
    final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, random_state=42)
    cluster_labels = final_kmeans.fit_predict(X_scaled)
    
    # Attach the cluster labels back to the original dataframe to see the groups
    df['Cluster_Group'] = cluster_labels
    
    print(f"    -> Successfully clustered {df.shape[0]} employees into {optimal_k} groups.")
    
    # Show how many employees fell into each cluster
    print("\n[+] Cluster Distribution:")
    print(df['Cluster_Group'].value_counts().sort_index())

    # ==========================================================
    # 5. VISUALIZATION USING PCA (Dimensionality Reduction)
    # ==========================================================
    print("\n[+] 5. Applying PCA to visualize multi-dimensional clusters in 2D...")
    
    # Squash all columns down into exactly 2 columns (X and Y coordinates)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    
    # Plot each cluster with a different color
    colors = ['#F44336', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
    for i in range(optimal_k):
        plt.scatter(X_pca[cluster_labels == i, 0], 
                    X_pca[cluster_labels == i, 1], 
                    s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7, edgecolors='k')
        
    # Plot the centroids (the center of each cluster)
    centroids_pca = pca.transform(final_kmeans.cluster_centers_)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                s=250, c='yellow', marker='*', edgecolor='black', label='Centroids')

    plt.title('2D PCA Visualization of Employee Salary Clusters')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION TO WRITE IN YOUR PRACTICAL SHEET:")
    print("1. Unsupervised learning requires no target variable; data was grouped purely by similarity.")
    print("2. The Elbow Method was used to plot inertia (WCSS), identifying the optimal k.")
    print("3. StandardScaler was mandatory to prevent large variables (Salary) from overpowering small ones (Experience).")
    print("4. PCA (Principal Component Analysis) was used to reduce the dataset to 2D for accurate visual plotting.")
    print("="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'Employee_Salary.csv'
employee_kmeans_clustering(file_name)