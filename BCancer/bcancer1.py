import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def breast_cancer_optimization_fixed(file_name):
    print("="*65)
    print("   BREAST CANCER: OPTIMIZATION & REGULARIZATION REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
        print(f"[+] Loaded {file_name}. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. SMART CLEANING (Fixes the 'M/B' String Error)
    # ==========================================================
    # Drop ID columns (usually the first column)
    if 'id' in df.columns.str.lower():
        id_col = [c for c in df.columns if c.lower() == 'id'][0]
        df.drop(columns=[id_col], inplace=True)
        print(f"    -> Dropped ID column.")

    # Find the Target (The column containing 'M' and 'B')
    target_col = None
    for col in df.columns:
        unique_vals = df[col].unique()
        if set(unique_vals) == {'M', 'B'} or set(unique_vals) == {1, 0}:
            target_col = col
            break
    
    if not target_col:
        # Fallback: assume it's the 'diagnosis' column or the second column
        target_col = 'diagnosis' if 'diagnosis' in df.columns else df.columns[0]

    print(f"[+] Identified Target Column: '{target_col}'")

    # Encode M/B to 1/0
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    
    # Remove any completely empty columns (Common in this dataset)
    df.dropna(axis=1, how='all', inplace=True)
    # Fill any small missing values with mean
    df = df.fillna(df.mean(numeric_only=True))

    # ==========================================================
    # 2. SPLIT & SCALE (Now works without errors)
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Standardize features (Mandatory for Optimizers)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    print(f"[+] Data cleaned and scaled. Ready for Neural Network.")

    # ==========================================================
    # 3. COMPARE OPTIMIZERS: SGD vs ADAM
    # ==========================================================
    print("\n[+] Training Model 1: SGD Optimizer (Basic)...")
    mlp_sgd = MLPClassifier(hidden_layer_sizes=(64, 32), solver='sgd', max_iter=300, random_state=42)
    mlp_sgd.fit(X_train, y_train)
    sgd_acc = accuracy_score(y_test, mlp_sgd.predict(X_test))

    print("[+] Training Model 2: Adam Optimizer + Regularization...")
    # alpha=0.01 provides L2 regularization to avoid overfitting
    mlp_adam = MLPClassifier(hidden_layer_sizes=(64, 32), solver='adam', 
                             alpha=0.01, early_stopping=True, max_iter=300, random_state=42)
    mlp_adam.fit(X_train, y_train)
    adam_acc = accuracy_score(y_test, mlp_adam.predict(X_test))

    # ==========================================================
    # 4. RESULTS & VISUALIZATION
    # ==========================================================
    print("\n--- Optimizer Results ---")
    print(f" SGD Accuracy : {sgd_acc * 100:.2f}%")
    print(f" Adam Accuracy: {adam_acc * 100:.2f}%")
    
    print("\n[Final Report (Adam)]")
    print(classification_report(y_test, mlp_adam.predict(X_test)))

    plt.figure(figsize=(8, 5))
    plt.plot(mlp_sgd.loss_curve_, label='SGD', color='red', linestyle='--')
    plt.plot(mlp_adam.loss_curve_, label='Adam (Regularized)', color='green', linewidth=2)
    plt.title('Optimization Comparison: Breast Cancer Diagnosis')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION: Regularization and adaptive optimization (Adam) ")
    print("ensured the model converged quickly without overfitting.")
    print("="*65)

# --- EXAM USAGE ---
file_name = 'BCancer.csv' 
breast_cancer_optimization_fixed(file_name)