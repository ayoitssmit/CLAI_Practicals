import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
warnings.filterwarnings('ignore')

def preprocess_telco_data(file_name):
    print("="*65)
    print("      TELCO CHURN: PREPROCESSING & SCALING REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. DROP IDENTIFIERS, GPS, AND DATA LEAKAGE COLUMNS
    # ==========================================================
    # We drop ID, high-cardinality strings (Lat Long), and post-churn metrics 
    cols_to_drop = ['CustomerID', 'Lat Long', 'Country', 'State', 'City', 
                    'Churn Reason', 'Churn Score', 'Churn Value', 'Count']
    
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_drop, inplace=True)
    print(f"[+] 1. Dropped irrelevant/leakage columns to prevent column explosion.")

    # ==========================================================
    # 2. HANDLE MISSING VALUES & THE 'TOTAL CHARGES' TRAP
    # ==========================================================
    print("\n[+] 2. Handling Missing Values...")
    
    if 'Total Charges' in df.columns:
        # Coerce forces blank spaces to become actual NaN values
        df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
        print("    -> Fixed 'Total Charges' data type from text to numeric.")

    # Impute missing numerical values with Median
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0 and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        print("    -> Imputed missing numerical values with the Median.")

    # ==========================================================
    # 3. ENCODING (Categorical to Numeric)
    # ==========================================================
    print("\n[+] 3. Encoding Categorical Variables...")
    
    # Label Encode the actual Target Column ('Churn Label' -> Yes/No to 1/0)
    target_col = 'Churn Label'
    if target_col in df.columns:
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col].astype(str))
        print(f"    -> Label Encoded Target Column '{target_col}'.")

    # One-Hot Encode remaining text columns (Gender, InternetService, etc.)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if target_col in cat_cols: 
        cat_cols.remove(target_col)
        
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Convert the new True/False dummy columns to 1/0 integers to keep it clean
        df = df.astype(float) 
        print(f"    -> Applied One-Hot Encoding to {len(cat_cols)} features.")

    # ==========================================================
    # 4. FEATURE SCALING
    # ==========================================================
    print("\n[+] 4. Feature Scaling...")
    
    # Scale only the original numerical columns
    num_cols = ['Zip Code', 'Latitude', 'Longitude', 'Tenure Months', 'Monthly Charges', 'Total Charges', 'CLTV']
    existing_num = [c for c in num_cols if c in df.columns]

    scaler = StandardScaler()
    df[existing_num] = scaler.fit_transform(df[existing_num])
    print(f"    -> Applied StandardScaler to numeric features.")

    # ==========================================================
    # FINAL OUTPUT
    # ==========================================================
    print("\n" + "="*65)
    print(" PREPROCESSING COMPLETE! DATASET IS READY FOR MODEL TRAINING.")
    print(f" Final Dataset Shape: {df.shape}")
    print("="*65)
    
    return df

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'Telco_Customer_Churn.csv'
processed_dataset = preprocess_telco_data(file_name)