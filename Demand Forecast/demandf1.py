import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def demand_mlp_classification(file_name):
    print("="*65)
    print("   DEMAND FORECASTING: MLP CLASSIFICATION REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. THE TRAP: CONVERTING DEMAND TO CATEGORIES
    # ==========================================================
    print("\n[+] 1. Target Engineering...")
    
    # Identify the Demand/Sales column
    target_col = None
    for col in df.columns:
        if col.lower() in ['demand', 'sales', 'units_sold', 'quantity', 'total_sales']:
            target_col = col
            break
    
    if not target_col:
        target_col = df.columns[-1]

    # If the target is continuous (numbers), convert to High/Low classes
    if df[target_col].dtype in ['float64', 'int64'] and df[target_col].nunique() > 10:
        median_val = df[target_col].median()
        df['Demand_Level'] = (df[target_col] >= median_val).astype(int)
        
        # Drop the original numerical demand to prevent cheating (Data Leakage)
        df.drop(columns=[target_col], inplace=True)
        target_col = 'Demand_Level'
        print(f"    -> Converted numerical demand to binary: 'High' (>= {median_val}) and 'Low'.")
    else:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))

    # ==========================================================
    # 2. TIME-SERIES FEATURE ENGINEERING (A+ Logic)
    # ==========================================================
    # Check for date columns and extract meaningful numbers
    for col in df.columns:
        if 'date' in col.lower() or df[col].dtype == 'object':
            try:
                date_series = pd.to_datetime(df[col])
                df[f'{col}_month'] = date_series.dt.month
                df[f'{col}_day'] = date_series.dt.day
                df[f'{col}_dayofweek'] = date_series.dt.dayofweek
                df.drop(columns=[col], inplace=True)
                print(f"[+] Extracted Month/Day features from '{col}'.")
            except:
                continue

    # ==========================================================
    # 3. PREPROCESSING: IMPUTE & ENCODE
    # ==========================================================
    # Drop IDs
    ids = ['id', 'order_id', 'store_id']
    df.drop(columns=[c for c in df.columns if c.lower() in ids], inplace=True, errors='ignore')

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median() if df[col].dtype != 'object' else df[col].mode()[0])

    # One-Hot Encode remaining categories
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in cat_cols: cat_cols.remove(target_col)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df = df.astype(float)

    # ==========================================================
    # 4. SPLIT & SCALE (Mandatory for MLP)
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    # ==========================================================
    # 5. TRAIN MLP CLASSIFIER
    # ==========================================================
    print("\n[+] Training Neural Network (MLPClassifier)...")
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), 
                        max_iter=500, 
                        early_stopping=True, # Prevents Overfitting
                        random_state=42)
    mlp.fit(X_train, y_train)
    
    y_pred = mlp.predict(X_test)
    print(f"\n--- Classification Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(classification_report(y_test, y_pred))

    # ==========================================================
    # 6. VISUALIZATION: LOSS CURVE
    # ==========================================================
    plt.figure(figsize=(7, 4))
    plt.plot(mlp.loss_curve_, color='red', label='Loss')
    plt.title('Demand Forecasting: MLP Training Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print("\n" + "="*65)

# --- EXAM USAGE ---
file_name = 'demand_forecasting.csv'
demand_mlp_classification(file_name)