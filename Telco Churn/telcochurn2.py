import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def perform_linear_regression(file_name):
    print("="*65)
    print("       TELCO DATASET: LINEAR REGRESSION REPORT")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. THE TRAP: SETTING A CONTINUOUS TARGET
    # ==========================================================
    # We CANNOT predict 'Churn' with Linear Regression. We must predict a number.
    # We will predict 'Total Charges'.
    target_col = 'Total Charges'
    
    if target_col not in df.columns:
        print(f"ERROR: Could not find the numerical target column '{target_col}'.")
        return
        
    print(f"[+] Task correctly identified as Regression. Target variable set to: '{target_col}'")

    # ==========================================================
    # 2. PREPROCESSING & CLEANING (Handling the 'Total Charges' blanks)
    # ==========================================================
    # Convert target to numeric (forces blank spaces to NaN)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # Drop rows where our target variable is missing (you cannot train on missing targets)
    df = df.dropna(subset=[target_col])
    
    # Drop IDs and Leakage columns
    cols_to_drop = ['CustomerID', 'Lat Long', 'Country', 'State', 'City', 
                    'Churn Reason', 'Churn Score', 'Churn Value', 'Count']
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=existing_drop)

    # Impute missing numerical features with Median
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # One-Hot Encode Categorical Variables
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    print("[+] Preprocessing, Encoding, and Cleaning complete.")

    # ==========================================================
    # 3. SPLIT & SCALE DATA
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scaling features (Crucial for good linear regression weights)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print(f"[+] Data split into Train and Test sets. (Train: {X_train.shape[0]} samples)")

    # ==========================================================
    # 4. MODEL TRAINING: LINEAR REGRESSION
    # ==========================================================
    print("\n[+] Training Linear Regression Model...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    # Predictions
    y_pred = lr_model.predict(X_test)

    # ==========================================================
    # 5. REGRESSION EVALUATION METRICS
    # ==========================================================
    # Note: Accuracy is for Classification. For Regression, we use R2, MSE, and MAE.
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Evaluation Results ---")
    print(f"    -> R-Squared (R2) Score : {r2:.4f} (Closer to 1.0 is better)")
    print(f"    -> Mean Absolute Error  : {mae:.2f}")
    print(f"    -> Root Mean Sq Error   : {rmse:.2f}")

    # ==========================================================
    # 6. VISUALIZATION (Actual vs Predicted)
    # ==========================================================
    print("\n[+] Generating Actual vs Predicted scatter plot...")
    plt.figure(figsize=(8, 6))
    
    # Scatter plot of actual vs predicted
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue', label='Data Points')
    
    # The perfect prediction line (y = x)
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], color='red', linewidth=2, linestyle='--', label='Perfect Prediction Line')
    
    plt.title('Linear Regression: Actual vs Predicted Total Charges')
    plt.xlabel('Actual Total Charges')
    plt.ylabel('Predicted Total Charges')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'Telco_Customer_Churn.csv'
perform_linear_regression(file_name)