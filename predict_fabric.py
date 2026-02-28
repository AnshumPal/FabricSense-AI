# predict_fabric.py
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('textile_classifier_rf.pkl')

# Map fab_id to fabric type
fab_map = {0: 'Fabric_Type_0', 1: 'Fabric_Type_1', 2: 'Fabric_Type_2'}

# Load new data
new_sample = pd.read_csv('new_fabric_sample.csv')  # replace with actual sample file

# Select top 100 bands
top_bands = [f'C{i:03d}' for i in range(113, 213)]  # same as training
X_new = new_sample[top_bands]

# Make predictions
preds = model.predict(X_new)
pred_labels = [fab_map[i] for i in preds]

# Show results
new_sample['Predicted_Fabric'] = pred_labels
print(new_sample[['Predicted_Fabric']])