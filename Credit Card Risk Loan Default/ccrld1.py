import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def credit_risk_classification(file_name):
    print("="*65)
    print(" CREDIT RISK: K-NN vs DECISION TREE CLASSIFICATION REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. DYNAMIC TARGET IDENTIFICATION
    # ==========================================================
    # Look for common credit risk target columns
    target_col = None
    common_targets = ['default', 'loan_status', 'status', 'risk', 'target', 'class']
    
    for col in df.columns:
        if col.lower() in common_targets:
            target_col = col
            break
            
    if not target_col:
        # Fallback: Assume the last column is the target (standard dataset structure)
        target_col = df.columns[-1]

    print(f"[+] Identified Classification Target: '{target_col}'")

    # ==========================================================
    # 2. PREPROCESSING & IMPUTATION
    # ==========================================================
    print("\n[+] Preprocessing Data...")
    
    # Drop irrelevant IDs (IDs ruin both k-NN distances and Tree splits)
    ids = ['id', 'customer_id', 'loan_id']
    df.drop(columns=[c for c in df.columns if c.lower() in ids], inplace=True, errors='ignore')

    # Drop missing targets
    df.dropna(subset=[target_col], inplace=True)

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median()) # Median for skewed money/income data
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Encode the Target variable (e.g., "Default" / "Not Default" -> 1 / 0)
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))

    # One-Hot Encode categorical features (Gender, Education, etc.)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df = df.astype(float) # Ensure everything is a clean number

    # ==========================================================
    # 3. SPLITTING & SCALING (THE EXAMINER'S TRAP)
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # TRAP AVOIDED: Stratify=y ensures the rare "Defaults" are split evenly between train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # TRAP AVOIDED: k-NN REQUIRES scaled data. We scale X_train and X_test.
    # Decision Trees don't technically need it, but using scaled data for both ensures a fair comparison.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"[+] Data Split & Scaled. (Stratified to handle credit risk imbalance)")

    # ==========================================================
    # 4. TRAIN & EVALUATE MODEL 1: K-NEAREST NEIGHBORS (k-NN)
    # ==========================================================
    print("\n" + "-"*40)
    print("   MODEL 1: K-NEAREST NEIGHBORS (k=5)")
    print("-" * 40)
    
    knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) # p=2 is Euclidean distance
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)
    
    knn_acc = accuracy_score(y_test, knn_pred)
    print(f"k-NN Accuracy: {knn_acc * 100:.2f}%")
    print(classification_report(y_test, knn_pred))

    # ==========================================================
    # 5. TRAIN & EVALUATE MODEL 2: DECISION TREE
    # ==========================================================
    print("\n" + "-"*40)
    print("   MODEL 2: DECISION TREE")
    print("-" * 40)
    
    # TRAP AVOIDED: Set max_depth to prevent the tree from overfitting
    dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_pred = dt.predict(X_test_scaled)
    
    dt_acc = accuracy_score(y_test, dt_pred)
    print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")
    print(classification_report(y_test, dt_pred))

    # ==========================================================
    # 6. VISUALIZATION: MODEL COMPARISON
    # ==========================================================
    print("\n[+] Generating Model Comparison Chart...")
    
    models = ['k-Nearest Neighbors', 'Decision Tree']
    accuracies = [knn_acc * 100, dt_acc * 100]
    
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, accuracies, color=['#2196F3', '#4CAF50'], edgecolor='black')
    
    # Add the percentage text on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.title('Credit Risk Classification: k-NN vs Decision Tree', fontsize=13)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(accuracies) + 15) # Add space for text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION FOR EXAMINER:")
    if dt_acc > knn_acc:
        print("-> Decision Tree performed better. It is often preferred in credit risk ")
        print("   because the split rules (e.g., Income > $50k) are easily interpretable by banks.")
    else:
        print("-> k-NN performed better. The mandatory StandardScaler ensured that large ")
        print("   loan values did not mathematically overwhelm smaller values like Age.")
    print("="*65)


# ==========================================
#               EXAM USAGE
# ==========================================

# ALL YOU DO IS CHANGE THIS EXACT FILE NAME TOMORROW:
file_name = 'credit_risk_dataset.csv'
credit_risk_classification(file_name)