import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocess_demand_forecasting(file_name):
    print("="*65)
    print("   DEMAND FORECASTING: PREPROCESSING & SCALING REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
        print(f"[+] Successfully loaded dataset. Initial Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. TIME-SERIES FEATURE ENGINEERING (Crucial for Demand)
    # ==========================================================
    # Automatically find date columns and extract features
    for col in df.columns:
        if 'date' in col.lower() or df[col].dtype == 'object':
            try:
                # Try to convert to datetime
                date_series = pd.to_datetime(df[col])
                df[f'{col}_month'] = date_series.dt.month
                df[f'{col}_day'] = date_series.dt.day
                df[f'{col}_dayofweek'] = date_series.dt.dayofweek
                df.drop(columns=[col], inplace=True)
                print(f"[+] Extracted Month, Day, and DayOfWeek from '{col}'.")
            except:
                continue # If it's not a date, just move on

    # ==========================================================
    # 2. DROP IRRELEVANT IDENTIFIERS (IDs)
    # ==========================================================
    # We drop IDs because they are unique to each row and have no predictive power.
    ids_to_drop = ['id', 'order_id', 'store_id', 'item_id', 'customer_id']
    existing_ids = [c for c in ids_to_drop if c.lower() in [col.lower() for col in df.columns]]
    df.drop(columns=existing_ids, inplace=True, errors='ignore')
    print(f"[+] Dropped identification columns: {existing_ids}")

    # ==========================================================
    # 3. HANDLE MISSING VALUES (HINT 1)
    # ==========================================================
    print("\n[+] Handling Missing Values...")
    missing_data = df.isnull().sum()
    if missing_data.any():
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].median()) # Median for numerical
                else:
                    df[col] = df[col].fillna(df[col].mode()[0]) # Mode for categorical
        print("    -> Missing values imputed (Median for numbers, Mode for text).")
    else:
        print("    -> No missing values found.")

    # ==========================================================
    # 4. ENCODING (HINT 2)
    # ==========================================================
    print("\n[+] Encoding Categorical Variables...")
    
    # Identify categorical columns (strings)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if len(cat_cols) > 0:
        # Check for target column (usually 'sales' or 'demand') to avoid encoding it
        target_candidates = ['sales', 'demand', 'quantity', 'units_sold']
        target_col = None
        for cand in target_candidates:
            if cand in [c.lower() for c in df.columns]:
                target_col = cand
                break
        
        # Apply One-Hot Encoding to categorical features
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Convert True/False to 1/0
        df = df.astype(float)
        print(f"    -> Applied One-Hot Encoding to {len(cat_cols)} features.")
    else:
        print("    -> No categorical text columns found to encode.")

    # ==========================================================
    # 5. FEATURE SCALING (HINT 3)
    # ==========================================================
    print("\n[+] Feature Scaling...")
    # Scale all columns except the target (demand/sales)
    # If the user didn't specify a target, we scale everything for general preprocessing
    scaler = StandardScaler()
    df_scaled_values = scaler.fit_transform(df)
    df_final = pd.DataFrame(df_scaled_values, columns=df.columns)
    print("    -> Applied StandardScaler to all features.")

    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================
    print("\n" + "="*65)
    print(" PREPROCESSING COMPLETE!")
    print(f" Final Dataset Shape: {df_final.shape}")
    print("="*65)
    
    print("\n[Preview of Processed Data (First 3 rows)]")
    print(df_final.head(3).round(3))
    
    return df_final

# ==========================================
#               EXAM USAGE
# ==========================================

# ALL YOU DO IS CHANGE THIS FILE NAME:
file_name = 'demand_forecasting.csv'
processed_df = preprocess_demand_forecasting(file_name)