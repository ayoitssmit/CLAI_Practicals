import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def diabetes_hyperparameter_tuning(file_name):
    print("="*65)
    print("   DIABETES RISK: HYPERPARAMETER TUNING (GRID SEARCH)")
    print("="*65)

    try:
        df = pd.read_csv(file_name)
        print(f"[+] Dataset Loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"ERROR: '{file_name}' not found.")
        return

    # ==========================================================
    # 1. PREPROCESSING & CLEANING
    # ==========================================================
    # Identify Target (usually 'class', 'outcome', or 'diabetes')
    target_col = None
    for col in df.columns:
        if col.lower() in ['class', 'outcome', 'diabetes', 'result', 'target']:
            target_col = col
            break
    if not target_col: target_col = df.columns[-1]

    # Encode Categorical Variables (Gender, Polyuria, etc.)
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].astype(str))

    # Handle Missing Values (Impute with Median)
    df = df.fillna(df.median())

    # ==========================================================
    # 2. SPLIT & SCALE
    # ==========================================================
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================================
    # 3. BASELINE MODEL (Default Settings)
    # ==========================================================
    print("\n[+] 1. Training Baseline Random Forest (Default)...")
    rf_base = RandomForestClassifier(random_state=42)
    rf_base.fit(X_train_scaled, y_train)
    base_acc = accuracy_score(y_test, rf_base.predict(X_test_scaled))
    print(f"    -> Baseline Accuracy: {base_acc * 100:.2f}%")

    # ==========================================================
    # 4. GRID SEARCH (Hyperparameter Tuning)
    # ==========================================================
    print("\n[+] 2. Running GridSearchCV (Searching for optimal params)...")
    
    # We define the 'Grid' of parameters to test
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'criterion': ['gini', 'entropy']
    }

    # GridSearchCV will try every combination of these parameters
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                               param_grid=param_grid,
                               cv=5,            # 5-fold cross validation
                               n_jobs=-1,       # Use all CPU cores for speed
                               scoring='accuracy',
                               verbose=1)
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Extract the best model
    best_model = grid_search.best_estimator_
    tuned_acc = accuracy_score(y_test, best_model.predict(X_test_scaled))

    # ==========================================================
    # 5. RESULTS & COMPARISON
    # ==========================================================
    print("\n" + "-"*40)
    print("   TUNING RESULTS")
    print("-" * 40)
    print(f" Best Parameters found:")
    for param, val in grid_search.best_params_.items():
        print(f"  -> {param}: {val}")
    
    print(f"\n Tuned Accuracy: {tuned_acc * 100:.2f}%")
    print("-" * 40)

    # ==========================================================
    # 6. VISUALIZATION
    # ==========================================================
    plt.figure(figsize=(6, 5))
    models = ['Baseline (Default)', 'Optimized (GridSearch)']
    scores = [base_acc * 100, tuned_acc * 100]
    
    bars = plt.bar(models, scores, color=['#607D8B', '#FF5722'], edgecolor='black', width=0.6)
    plt.ylabel('Accuracy (%)')
    plt.title('Impact of Hyperparameter Tuning on Diabetes Risk Model')
    
    # Label the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}%', ha='center', fontweight='bold')

    plt.ylim(min(scores)-5, max(scores)+5)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION:")
    print("1. Used GridSearchCV with 5-fold Cross-Validation to optimize parameters.")
    print("2. Tested variations of tree depth and split criteria.")
    print("3. Hyperparameter tuning ensures the model generalizes well and ")
    print("   doesn't just overfit the training data.")
    print("="*65)

# --- EXAM USAGE ---
file_name = 'diabetes_risk.csv'
diabetes_hyperparameter_tuning(file_name)