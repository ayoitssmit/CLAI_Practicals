import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def student_mlp_classification(file_name):
    print("="*65)
    print("   STUDENT PERFORMANCE: MLP CLASSIFICATION REPORT")
    print("="*65)

    try:
        # Handle comma or semicolon separated files dynamically
        with open(file_name, 'r') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        df = pd.read_csv(file_name, sep=sep)
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. THE TRAP: ENGINEERING A CLASSIFICATION TARGET
    # ==========================================================
    print("\n[+] 1. Target Engineering (Handling the Classification Trap)...")
    
    # Check for score columns to create a 'Pass/Fail' target
    score_cols = [c for c in df.columns if 'score' in c.lower() or 'grade' in c.lower()]
    
    if len(score_cols) > 0:
        # Calculate Average Score
        df['Average_Score'] = df[score_cols].mean(axis=1)
        # Create Target: 1 if Pass (>= 60), 0 if Fail (< 60)
        df['Pass_Exam'] = (df['Average_Score'] >= 60).astype(int)
        target_col = 'Pass_Exam'
        
        # CRITICAL: Drop the original scores to prevent Data Leakage!
        # If the model knows the math score, predicting 'Pass' is cheating.
        df = df.drop(columns=score_cols + ['Average_Score'])
        print(f"    -> Converted continuous scores into Categorical Target: '{target_col}'")
        print(f"    -> Dropped original score columns to prevent Data Leakage.")
    else:
        # Fallback: Just grab the last categorical column if no scores are found
        target_col = df.select_dtypes(include=['object', 'string', 'category']).columns[-1]
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
        print(f"    -> Using existing categorical column as target: '{target_col}'")

    # ==========================================================
    # 2. PREPROCESSING: ENCODING & IMPUTATION
    # ==========================================================
    print("\n[+] 2. Preprocessing & Encoding...")
    
    # Drop IDs
    identifiers = ['student_id', 'ID', 'roll_no']
    existing_ids = [c for c in identifiers if c in df.columns]
    df.drop(columns=existing_ids, inplace=True, errors='ignore')

    # Drop missing target rows
    df = df.dropna(subset=[target_col])

    # Impute missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # One-Hot Encode Categorical Features
    cat_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
        
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        df = df.astype(float) # Ensure clean numbers
        print(f"    -> Applied One-Hot Encoding to {len(cat_cols)} text columns.")

    # ==========================================================
    # 3. SPLIT & SCALE DATA (CRUCIAL FOR NEURAL NETWORKS)
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # NNs are highly sensitive to unscaled data. StandardScaler is mandatory.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\n[+] Data Split: {X_train.shape[0]} Training Samples, {X_test.shape[0]} Testing Samples")

    # ==========================================================
    # 4. TRAIN MLP CLASSIFIER (NEURAL NETWORK)
    # ==========================================================
    print("\n[+] Training Multi-Layer Perceptron (Neural Network)...")
    
    # 2 hidden layers (100 neurons, then 50 neurons), Early stopping to prevent overfitting
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50), 
                        activation='relu', 
                        solver='adam', 
                        max_iter=500, 
                        early_stopping=True,
                        random_state=42)
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # ==========================================================
    # 5. EVALUATION METRICS
    # ==========================================================
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "-"*45)
    print("   NEURAL NETWORK EVALUATION RESULTS")
    print("-"*45)
    print(f" Test Accuracy: {acc * 100:.2f}%\n")
    print(" Detailed Classification Report:")
    
    # Dynamic target names based on what we predicted
    if target_col == 'Pass_Exam':
        target_names = ['Fail (<60)', 'Pass (>=60)']
    else:
        target_names = None
        
    print(classification_report(y_test, y_pred, target_names=target_names))
    print("-"*45)

    # ==========================================================
    # 6. VISUALIZATION: MLP LOSS CURVE
    # ==========================================================
    print("\n[+] Generating Neural Network Loss Curve...")
    plt.figure(figsize=(7, 5))
    
    plt.plot(mlp.loss_curve_, color='#673AB7', linewidth=2.5, label='Training Loss')
    plt.title('MLP Neural Network: Training Loss Curve', fontsize=14)
    plt.xlabel('Epochs (Iterations)', fontsize=12)
    plt.ylabel('Loss (Error)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

# Just change this file name tomorrow!
file_name = 'StudentsPerformance.csv' 
student_mlp_classification(file_name)