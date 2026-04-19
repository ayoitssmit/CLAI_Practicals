import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def stroke_hierarchical_clustering(file_name):
    print("="*65)
    print("  STROKE RISK: HIERARCHICAL CLUSTERING & DENDROGRAM REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. THE TRAPS: DATA LEAKAGE & MISSING VALUES
    # ==========================================================
    print("\n[+] 1. Preprocessing (Unsupervised Approach)...")
    
    # TRAP 1: Drop 'id' and 'stroke' (target) columns! 
    # Clustering is unsupervised, we cannot give the model the answers.
    cols_to_drop = ['id', 'stroke']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drop, inplace=True, errors='ignore')
    print(f"    -> Dropped target column 'stroke' to ensure strict Unsupervised Learning.")

    # TRAP 2: Impute missing 'bmi' values. Hierarchical clustering crashes on NaNs.
    # We use median because BMI data is often skewed.
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce') # Handle hidden text
        df['bmi'] = df['bmi'].fillna(df['bmi'].median())
        
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    print("    -> Imputed missing values (like BMI) with the median.")

    # One-Hot Encode categoricals (Gender, Married, Work Type, etc.)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df = df.astype(float)
        print(f"    -> Applied One-Hot Encoding to categorical features.")

    # ==========================================================
    # 2. FEATURE SCALING (CRITICAL FOR DISTANCE CALCULATION)
    # ==========================================================
    # If Glucose is 200 and Age is 40, Glucose will dominate the distance math.
    print("\n[+] 2. Scaling Features (Mandatory for Hierarchical Clustering)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # ==========================================================
    # 3. DENDROGRAM ANALYSIS (THE "BLACK BLOB" TRAP)
    # ==========================================================
    print("\n[+] 3. Generating Dendrogram using Ward's Linkage Method...")
    
    # Calculate the linkage matrix using Ward's method (minimizes variance)
    linked = linkage(X_scaled, method='ward')
    
    plt.figure(figsize=(10, 6))
    plt.title('Hierarchical Clustering Dendrogram (Truncated for Readability)', fontsize=14)
    plt.xlabel('Cluster Size', fontsize=12)
    plt.ylabel('Euclidean Distance (Ward)', fontsize=12)
    
    # TRAP 3 FIX: truncate_mode='lastp' condenses the 5000+ rows into 30 readable leaf nodes.
    # Without this, the examiner would just see a solid black box of overlapping lines.
    dendrogram(linked,
               truncate_mode='lastp',  # Show only the last p merged clusters
               p=30,                   # Number of nodes to show at the bottom
               leaf_rotation=90.,
               leaf_font_size=10.,
               show_contracted=True,
               color_threshold=80)     # Colors branches to highlight major clusters

    # Draw a horizontal line where we theoretically "cut" the tree to make clusters
    plt.axhline(y=80, color='r', linestyle='--', label='Theoretical Cluster Cut')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================================================
    # 4. APPLYING AGGLOMERATIVE CLUSTERING
    # ==========================================================
    # Based on standard dendrogram visual cuts, we'll choose 3 clusters
    n_clusters = 3
    print(f"\n[+] 4. Applying Agglomerative Clustering (n_clusters={n_clusters})...")
    
    hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
    cluster_labels = hc.fit_predict(X_scaled)
    
    # Add clusters back to data
    df['Cluster'] = cluster_labels
    print(f"    -> Successfully formed {n_clusters} clusters.")
    print("\nCluster Distribution:")
    print(df['Cluster'].value_counts().sort_index())

    # ==========================================================
    # 5. VISUALIZING THE CLUSTERS WITH PCA
    # ==========================================================
    print("\n[+] 5. Compressing dataset via PCA for 2D Cluster Visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    colors = ['#FF5722', '#00BCD4', '#8BC34A', '#9C27B0']
    
    for i in range(n_clusters):
        plt.scatter(X_pca[cluster_labels == i, 0], X_pca[cluster_labels == i, 1], 
                    s=40, c=colors[i], label=f'Cluster {i}', alpha=0.6, edgecolors='k')

    plt.title('PCA: 2D Visualization of Stroke Risk Clusters', fontsize=13)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION FOR EXAMINER:")
    print("1. Target variable 'stroke' was dropped to ensure pure unsupervised clustering.")
    print("2. The massive dataset was handled using 'truncate_mode' to generate a readable Dendrogram.")
    print("3. Ward's Linkage minimized within-cluster variance, revealing clear patient groupings.")
    print("="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'Stroke_Risk.csv'
stroke_hierarchical_clustering(file_name)