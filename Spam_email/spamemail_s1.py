import pandas as pd
import matplotlib.pyplot as plt

def perform_full_eda(df):
    """
    Complete EDA and Profiling function. 
    Handles basic stats, text analysis (if applicable), and visualizations.
    """
    print("="*60)
    print("           FULL EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("="*60)
    
    # 1. Basic Profiling (Hints: dimensions, features, dtypes, missing)
    print(f"\n[+] Dataset Dimensions: {df.shape[0]} Rows, {df.shape[1]} Columns")
    print(f"[+] Feature Names: {df.columns.tolist()}")
    
    print("\n[+] Data Types:")
    print(df.dtypes)
    
    print("\n[+] Missing Values:")
    missing_data = df.isnull().sum()
    print(missing_data[missing_data > 0] if missing_data.any() else "No missing values!")

    # 2. Advanced Text EDA (Specific for your Spam dataset)
    # Automatically finds the text column (column with longest strings) and target column
    text_col = None
    target_col = None
    
    for col in df.columns:
        if df[col].nunique() < 10: 
            target_col = col  # The 'Category' column (ham/spam)
        elif df[col].dtype == 'object':
            text_col = col    # The 'Masseges' column

    if text_col and target_col:
        print(f"\n[+] Text Data Analysis:")
        # Create a new feature: Length of the message
        df['Message_Length'] = df[text_col].apply(lambda x: len(str(x)))
        print(f"Created new feature: 'Message_Length'")
        
        # Compare average length of Spam vs Ham
        print("\nAverage Message Length by Category:")
        print(df.groupby(target_col)['Message_Length'].mean().round(2))

    # 3. Visualization (Examiners LOVE graphs for EDA)
    if target_col:
        print("\n[+] Generating Class Imbalance Visualization...")
        
        # Plotting the target distribution
        df[target_col].value_counts().plot(kind='bar', color=['#4CAF50', '#F44336'])
        plt.title('Class Distribution (Ham vs Spam)')
        plt.xlabel('Category')
        plt.ylabel('Number of Messages')
        plt.xticks(rotation=0)
        
        # Show the plot in a popup window
        plt.tight_layout()
        plt.show()

    print("\n" + "="*60)

# ==========================================
#               EXAM USAGE
# ==========================================

file_name = 'spam mail.csv'

try:
    df = pd.read_csv(file_name)
    perform_full_eda(df)
except Exception as e:
    print(f"ERROR: {e}")