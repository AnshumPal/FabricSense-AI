# model_validation.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving the final model

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv('combined_dataset.csv')  # use your combined dataset file

# Use only top 100 important bands
top_bands = [f"C{i:03}" for i in range(100)]
X = df[top_bands].values
y = df['fab_id'].values

# -------------------------------
# K-Fold Validation
# -------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_confusions = []
fold_accuracies = []

print("🔹 Performing K-Fold Cross Validation...\n")

for i, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = model.score(X_test, y_test)
    fold_accuracies.append(acc)

    print(f"✅ Fold {i} Accuracy: {acc:.4f}")
    print(f"Fold {i} Classification Report:\n{classification_report(y_test, y_pred)}")

    fold_confusions.append(confusion_matrix(y_test, y_pred))

# -------------------------------
# Average Accuracy & Confusion Matrix
# -------------------------------
print(f"\nAverage Accuracy across folds: {np.mean(fold_accuracies):.4f}")

# Aggregate confusion matrices
agg_conf_matrix = sum(fold_confusions)
plt.figure(figsize=(6,5))
sns.heatmap(agg_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Aggregated Confusion Matrix (K-Fold)")
plt.show()

# -------------------------------
# Train final model on full dataset
# -------------------------------
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X, y)

# Save the trained model
joblib.dump(final_model, 'textile_rf_model.pkl')
print("\n🎯 Final model trained on full dataset and saved as 'textile_rf_model.pkl'.")