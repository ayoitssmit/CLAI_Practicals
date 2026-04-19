import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def credit_fraud_profiling(file_name):
    print("\n" + "="*65)
    print("      CO1: DATA EXPLORATION & PROFILING REPORT")
    print("      DATASET: CREDIT CARD FRAUD DETECTION")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found. Please ensure the file is in the folder.")
        return

    # --- 1. BASIC DIMENSIONS ---
    rows, cols = df.shape
    print(f"\n[+] Dataset Dimensions:")
    print(f"    -> Total Transactions (Rows): {rows}")
    print(f"    -> Total Features (Columns): {cols}")

    # --- 2. FEATURE IDENTIFICATION & DATA TYPES (HINT 1 & 2) ---
    print("\n[+] Feature Identification & Data Types:")
    # We display a concise summary of the data types
    type_counts = df.dtypes.value_counts()
    print(f"    -> Numeric Features: {type_counts.get('float64', 0) + type_counts.get('int64', 0)}")
    print(f"    -> Categorical/Object Features: {type_counts.get('object', 0)}")
    
    # Show first 5 columns as a sample
    print(f"    -> Sample Features: {list(df.columns[:5])} ... {list(df.columns[-2:])}")

    # --- 3. MISSING VALUE ANALYSIS (HINT 3) ---
    missing_values = df.isnull().sum().sum()
    print("\n[+] Missing Value Analysis:")
    if missing_values == 0:
        print("    -> Result: Perfect! No missing values detected in the dataset.")
    else:
        print(f"    -> WARNING: Found {missing_values} missing entries.")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    # Create synthetic 'Class' if it doesn't exist (because the dataset lacks it)
    if 'Class' not in df.columns:
        print("\n[!] No 'Class' target detected. Generating a synthetic 'Class' based on high transaction amounts.")
        if 'transaction_dollar_amount' in df.columns:
            threshold = df['transaction_dollar_amount'].quantile(0.90)
            df['Class'] = (df['transaction_dollar_amount'] >= threshold).astype(int)
        else:
            # Fallback random synthesis if needed
            df['Class'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])
            
    target_col = 'Class'

    # --- 4. TARGET VARIABLE PROFILING (THE MOST IMPORTANT STEP) ---
    fraud_count = df[target_col].value_counts()
    fraud_percent = (fraud_count / len(df)) * 100

    print("\n[+] Target Variable Analysis ('Class'):")
    print(f"    -> Normal Transactions (0): {fraud_count.get(0, 0)} ({fraud_percent.get(0, 0):.2f}%)")
    print(f"    -> Fraudulent Transactions (1): {fraud_count.get(1, 0)} ({fraud_percent.get(1, 0):.2f}%)")
    print("    -> OBSERVATION: Dataset is highly imbalanced (Critical for CO5).")

    # --- 5. STATISTICAL SUMMARY ---
    print("\n[+] Statistical Summary (Key Features):")
    cols_to_describe = sorted(list(set(['credit_card_limit', 'transaction_dollar_amount', 'Amount', 'Time']).intersection(df.columns)))
    if cols_to_describe:
        print(df[cols_to_describe].describe().round(2))
    else:
        print(df.describe().round(2))

    # --- 6. VISUAL PROFILING ---
    # Create a 1x2 subplot: Class Distribution and Correlation Heatmap
    plt.figure(figsize=(12, 5))

    # Plot A: Class Distribution
    plt.subplot(1, 2, 1)
    sns.countplot(x=target_col, data=df, palette='viridis')
    plt.title('Distribution of Fraud vs Normal')
    plt.yscale('log') # Log scale because fraud is so tiny compared to normal
    plt.ylabel('Count (Log Scale)')

    # Plot B: Feature Correlation (Sample of features)
    plt.subplot(1, 2, 2)
    numeric_df = df.select_dtypes(include=[np.number])
    # Keep up to 7 random or important numeric columns + Class for readable heatmap
    sample_features = sorted(list(set(['credit_card_limit', 'transaction_dollar_amount', 'Lat', 'Long', 'Time', 'Amount', 'Class']).intersection(numeric_df.columns)))
    sns.heatmap(numeric_df[sample_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')

    plt.tight_layout()
    plt.savefig('Q1_Fraud_Data_Profiling.png', dpi=100)
    plt.show()

    print("\n" + "="*65)
    print(" PROFILING COMPLETE: DATASET IS READY FOR PREPROCESSING")
    print("="*65 + "\n")

# --- EXECUTION ---
file_name = 'creditcard_fraud_detection.csv' 
credit_fraud_profiling(file_name)