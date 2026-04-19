import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def optimization_and_regularization(df):
    print("="*65)
    print("       OPTIMIZATION & REGULARIZATION TRAINING REPORT")
    print("="*65)

    # 1. Smart Target Detection (Assumes target is the last column, or has < 10 unique values)
    target_col = None
    if df.iloc[:, -1].nunique() < 10:
        target_col = df.columns[-1]
    else:
        for col in df.columns:
            if df[col].nunique() < 10:
                target_col = col
                break
                
    if not target_col:
        print("ERROR: Could not find a target/label column.")
        return
        
    print(f"[+] Detected Target Column: '{target_col}'")
    
    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    X_raw = df.drop(columns=[target_col])
    
    # 2. Smart Feature Detection (Text vs Numerical)
    text_cols = X_raw.select_dtypes(include=['object', 'string']).columns
    
    if len(text_cols) > 0 and X_raw[text_cols[0]].nunique() > 20:
        print(f"[+] Detected Text Dataset. Applying TF-IDF Vectorization on '{text_cols[0]}'...")
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1500)
        X = vectorizer.fit_transform(X_raw[text_cols[0]])
    else:
        print(f"[+] Detected Numerical Dataset. Applying Standard Scaling...")
        # Keep only numeric columns just to be safe
        X_num = X_raw.select_dtypes(include=['number'])
        
        # Fill any missing values with 0 before scaling to avoid crashes
        X_num = X_num.fillna(0) 
        X = StandardScaler().fit_transform(X_num)

    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================================
    # REQUIREMENT 1 & 2: Compare Optimizers & Avoid Overfitting
    # ==========================================================
    
    print("\n[+] Training Model 1: SGD Optimizer (Standard approach)")
    # Model 1 uses standard Stochastic Gradient Descent
    mlp_sgd = MLPClassifier(hidden_layer_sizes=(50,), solver='sgd', 
                            alpha=0.0001, max_iter=200, random_state=42)
    mlp_sgd.fit(X_train, y_train)
    sgd_acc = accuracy_score(y_test, mlp_sgd.predict(X_test))
    print(f"    -> SGD Test Accuracy: {sgd_acc * 100:.2f}%")

    print("\n[+] Training Model 2: ADAM Optimizer + Regularization (Avoids Overfitting)")
    # Model 2 uses Adam, strong L2 Regularization (alpha), and Early Stopping
    mlp_adam = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', 
                             alpha=0.01,           # L2 Regularization
                             early_stopping=True,  # Avoids Overfitting by stopping early
                             validation_fraction=0.1, 
                             max_iter=200, random_state=42)
    mlp_adam.fit(X_train, y_train)
    adam_acc = accuracy_score(y_test, mlp_adam.predict(X_test))
    print(f"    -> ADAM Test Accuracy: {adam_acc * 100:.2f}%")

    # ==========================================================
    # VISUALIZATION: Loss Curve
    # ==========================================================
    print("\n[+] Generating Optimizer Loss Curve Comparison...")
    plt.figure(figsize=(8, 5))
    
    plt.plot(mlp_sgd.loss_curve_, label='SGD Optimizer', color='red')
    plt.plot(mlp_adam.loss_curve_, label='Adam Optimizer + Regularized', color='green', linewidth=2)
    
    plt.title('Optimization Comparison: Training Loss Curve')
    plt.xlabel('Epochs (Iterations)')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*65)
    print("CONCLUSION TO WRITE IN YOUR PRACTICAL SHEET:")
    print("1. Adam optimizer was compared against standard SGD.")
    print("2. Overfitting was prevented by applying L2 Regularization (alpha=0.01) ")
    print("   and Early Stopping, which halts training when validation loss stops improving.")
    print("="*65)

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'spam mail.csv'

try:
    # Adding specific encoding just in case the CSV has weird characters
    df = pd.read_csv(file_name, encoding='latin-1') 
    optimization_and_regularization(df)
except FileNotFoundError:
    print(f"ERROR: The file '{file_name}' was not found in the current folder.")