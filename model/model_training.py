import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
df = pd.read_csv("combined_dataset.csv")

# ----------------------------
# 2️⃣ Define Features & Target
# ----------------------------
X = df.iloc[:, :224]     # spectral bands
y = df["fab_id"]         # target

print("Number of classes:", y.nunique())
print("Class distribution:\n", y.value_counts())

# ----------------------------
# 3️⃣ Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# 4️⃣ Scaling
# ----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----------------------------
# 5️⃣ Train Random Forest
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ----------------------------
# 6️⃣ Evaluate
# ----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 7️⃣ Feature Importance
# ----------------------------
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Band": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Bands:")
print(importance_df.head(10))




# Select top 30 bands
top_bands = importance_df.head(100)["Band"].values

X_reduced = df[top_bands]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reduced, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)

model_r = RandomForestClassifier(n_estimators=200, random_state=42)
model_r.fit(X_train_r, y_train_r)

y_pred_r = model_r.predict(X_test_r)

print("\nAccuracy using Top 100 Bands:", accuracy_score(y_test_r, y_pred_r))



from sklearn.svm import SVC

print("\nTraining SVM...")

svm_model = SVC(kernel="rbf", C=10, gamma="scale")

svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))