import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def student_linear_regression(file_name):
    print("="*65)
    print("    STUDENT PERFORMANCE: LINEAR REGRESSION REPORT")
    print("="*65)

    try:
        # Some student datasets use semicolons instead of commas, this handles both automatically
        with open(file_name, 'r') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        df = pd.read_csv(file_name, sep=sep)
        print(f"[+] Loaded dataset successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. DYNAMICALLY IDENTIFY THE TARGET COLUMN
    # ==========================================================
    # We look for common grade-related columns. If none are found, we take the last numerical column.
    target_col = None
    common_targets = ['G3', 'Score', 'Marks', 'Final_Grade', 'CGPA', 'Grade']
    
    for col in df.columns:
        if col.strip() in common_targets or col.lower() in [t.lower() for t in common_targets]:
            target_col = col
            break
            
    if not target_col:
        # Fallback: Assume the last numeric column is the target
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            target_col = num_cols[-1]

    if not target_col:
        print("ERROR: Could not identify a numerical target variable for Regression.")
        return
        
    print(f"[+] Identified Regression Target Variable: '{target_col}'")

    # ==========================================================
    # 2. DATA PREPROCESSING & CLEANING
    # ==========================================================
    # Drop irrelevant identifiers that shouldn't be learned
    identifiers = ['student_id', 'ID', 'roll_no']
    existing_ids = [c for c in identifiers if c in df.columns]
    df.drop(columns=existing_ids, inplace=True, errors='ignore')

    # Drop rows where the target itself is missing
    df = df.dropna(subset=[target_col])

    # Impute missing values for features
    print("\n[+] Cleaning and Imputing Missing Values...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())  # Median for numbers
            else:
                df[col] = df[col].fillna(df[col].mode()[0]) # Mode for text

    # One-Hot Encoding for categorical student data (Gender, Parent Job, etc.)
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        # Ensure dummy columns are converted to standard float
        df = df.astype(float)
        print(f"    -> Applied One-Hot Encoding to {len(cat_cols)} text columns.")

    # ==========================================================
    # 3. SPLIT & SCALE DATA
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"\n[+] Data Split: {X_train.shape[0]} Training Samples, {X_test.shape[0]} Testing Samples")

    # ==========================================================
    # 4. TRAIN THE LINEAR REGRESSION MODEL
    # ==========================================================
    print("\n[+] Training Linear Regression Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)

    # ==========================================================
    # 5. REGRESSION EVALUATION METRICS
    # ==========================================================
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n" + "-"*40)
    print("   MODEL EVALUATION RESULTS")
    print("-"*40)
    print(f" R-Squared (R2) Score : {r2:.4f}")
    print(f" Mean Absolute Error  : {mae:.4f}")
    print(f" Root Mean Sq Error   : {rmse:.4f}")
    print("-"*40)

    # Print the regression equation coefficients (Examiners love this!)
    print(f"\n[+] Top 3 Most Important Features (Largest Coefficients):")
    # Match coefficients to feature names, sort by absolute value
    coef_dict = dict(zip(X.columns, lr_model.coef_))
    sorted_coefs = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    for feature, coef in sorted_coefs:
        print(f"    -> {feature}: {coef:.4f}")

    # ==========================================================
    # 6. VISUALIZATION: Actual vs Predicted
    # ==========================================================
    print("\n[+] Generating Actual vs Predicted scatter plot...")
    plt.figure(figsize=(8, 6))
    
    plt.scatter(y_test, y_pred, alpha=0.6, color='#2196F3', edgecolor='k', s=50, label='Student Grades')
    
    # Perfect prediction diagonal line
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    plt.plot([min_val, max_val], [min_val, max_val], color='#F44336', linewidth=2, linestyle='--', label='Perfect Prediction')
    
    plt.title(f'Linear Regression: Actual vs Predicted {target_col}')
    plt.xlabel(f'Actual {target_col}')
    plt.ylabel(f'Predicted {target_col}')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

# Just change this file name tomorrow!
file_name = 'StudentsPerformance.csv' 
student_linear_regression(file_name)