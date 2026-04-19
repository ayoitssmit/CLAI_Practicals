import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               AdaBoostClassifier, VotingClassifier)
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Load & Sample ──────────────────────────────────────────────────────────────
df_full = pd.read_csv('creditcard_fraud_detection.csv')
df = df_full.sample(n=min(5000, len(df_full)), random_state=42).reset_index(drop=True)

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - ENSEMBLE METHODS")
print("=" * 60)
print(f"\nFull Dataset Shape : {df_full.shape}")
print(f"Sampled for Demo   : {df.shape}")

# ── Feature Engineering ───────────────────────────────────────────────────────
df['date'] = pd.to_datetime(df['date'])
df['hour']      = df['date'].dt.hour
df['month']     = df['date'].dt.month
df['dayofweek'] = df['date'].dt.dayofweek
df.drop(columns=['date', 'credit_card'], inplace=True)

# Fraud label: transaction > 80% of credit limit
threshold = df_full['transaction_dollar_amount'].quantile(0.90)
df['fraud'] = (df['transaction_dollar_amount'] >= threshold).astype(int)
print(f"\nFraud Label Distribution:\n{df['fraud'].value_counts()}")
print(f"Fraud % : {df['fraud'].mean()*100:.2f}%")

# ── Encode & Scale ────────────────────────────────────────────────────────────
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=['fraud'])
y = df['fraud']
print(f"\nFeatures: {list(X.columns)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ── 1. Baseline: Decision Tree ─────────────────────────────────────────────────
print("\n--- 1. Baseline: Decision Tree ---")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt  = accuracy_score(y_test, y_pred_dt)
auc_dt  = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
print(f"  Accuracy: {acc_dt:.4f}  |  AUC-ROC: {auc_dt:.4f}")

# ── 2. Random Forest ──────────────────────────────────────────────────────────
print("\n--- 2. Random Forest (n=100) ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf  = accuracy_score(y_test, y_pred_rf)
auc_rf  = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
cv_rf   = cross_val_score(rf, X_scaled, y, cv=5, scoring='roc_auc').mean()
print(f"  Accuracy: {acc_rf:.4f}  |  AUC-ROC: {auc_rf:.4f}  |  5-Fold CV AUC: {cv_rf:.4f}")

# ── 3. AdaBoost ───────────────────────────────────────────────────────────────
print("\n--- 3. AdaBoost (n=100) ---")
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
acc_ada = accuracy_score(y_test, y_pred_ada)
auc_ada = roc_auc_score(y_test, ada.predict_proba(X_test)[:, 1])
print(f"  Accuracy: {acc_ada:.4f}  |  AUC-ROC: {auc_ada:.4f}")

# ── 4. Gradient Boosting ──────────────────────────────────────────────────────
print("\n--- 4. Gradient Boosting (n=100) ---")
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                 max_depth=3, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb  = accuracy_score(y_test, y_pred_gb)
auc_gb  = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
print(f"  Accuracy: {acc_gb:.4f}  |  AUC-ROC: {auc_gb:.4f}")

# ── 5. Voting Ensemble ────────────────────────────────────────────────────────
print("\n--- 5. Soft Voting Ensemble (RF + AdaBoost + GB) ---")
voting = VotingClassifier(estimators=[
    ('rf',  RandomForestClassifier(n_estimators=100, random_state=42)),
    ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
    ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42))
], voting='soft')
voting.fit(X_train, y_train)
y_pred_vote = voting.predict(X_test)
acc_vote = accuracy_score(y_test, y_pred_vote)
auc_vote = roc_auc_score(y_test, voting.predict_proba(X_test)[:, 1])
print(f"  Accuracy: {acc_vote:.4f}  |  AUC-ROC: {auc_vote:.4f}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n--- Ensemble Comparison Summary ---")
print(f"  {'Model':<28} {'Accuracy':>10} {'AUC-ROC':>10}")
print(f"  {'-'*50}")
results = [('Decision Tree (Baseline)', acc_dt, auc_dt),
           ('Random Forest',           acc_rf, auc_rf),
           ('AdaBoost',                acc_ada, auc_ada),
           ('Gradient Boosting',       acc_gb, auc_gb),
           ('Voting Ensemble',         acc_vote, auc_vote)]
for name, acc, auc in results:
    print(f"  {name:<28} {acc:>10.4f} {auc:>10.4f}")

# Feature importance from RF
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(f"\nTop Feature Importances (Random Forest):\n{fi}")

# ── Visualizations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Credit Card Fraud - Ensemble Methods', fontsize=15)

# Accuracy & AUC bar comparison
names_short = ['DT\n(Base)', 'Random\nForest', 'Ada\nBoost', 'Gradient\nBoosting', 'Voting\nEnsemble']
accs = [r[1] for r in results]
aucs = [r[2] for r in results]
x = np.arange(len(names_short))
w = 0.35
axes[0, 0].bar(x - w/2, accs, w, label='Accuracy', color='steelblue')
axes[0, 0].bar(x + w/2, aucs, w, label='AUC-ROC',  color='coral')
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(names_short, fontsize=9)
axes[0, 0].set_ylim(0.5, 1.0)
axes[0, 0].legend()

# Confusion Matrix - Best model (Gradient Boosting)
best_idx = aucs.index(max(aucs))
best_name = results[best_idx][0]
cm = confusion_matrix(y_test, y_pred_gb)
ConfusionMatrixDisplay(cm, display_labels=['Non-Fraud', 'Fraud']).plot(
    ax=axes[0, 1], colorbar=False)
axes[0, 1].set_title(f'Confusion Matrix - Gradient Boosting')

# Feature Importance
fi.plot(kind='barh', ax=axes[1, 0], color='teal', edgecolor='black')
axes[1, 0].set_title('Random Forest - Feature Importance')

# GB: n_estimators vs accuracy
n_est_range = [10, 25, 50, 100]
gb_accs = []
for n in n_est_range:
    gb_n = GradientBoostingClassifier(n_estimators=n, random_state=42)
    gb_n.fit(X_train, y_train)
    gb_accs.append(accuracy_score(y_test, gb_n.predict(X_test)))
axes[1, 1].plot(n_est_range, gb_accs, 'g-o', linewidth=2)
axes[1, 1].set_title('Gradient Boosting: n_estimators vs Accuracy')
axes[1, 1].set_xlabel('n_estimators')
axes[1, 1].set_ylabel('Accuracy')

plt.tight_layout()
plt.savefig('Q10_CreditCard_Ensemble.png', dpi=100)
plt.show()
print("\nPlot saved as Q10_CreditCard_Ensemble.png")