# final_model_training.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load your combined dataset
df = pd.read_csv('combined_dataset.csv')  # path to your combined CSV

# Define features (top 100 bands)
top_bands = [f'C{i:03d}' for i in range(113, 213)]  # example top 100 from your previous importance
X = df[top_bands]
y = df['fab_id']

# Train final model on full dataset
final_model = RandomForestClassifier(n_estimators=200, random_state=42)
final_model.fit(X, y)

# Save the trained model
joblib.dump(final_model, 'textile_classifier_rf.pkl')
print("✅ Final model trained and saved successfully!")