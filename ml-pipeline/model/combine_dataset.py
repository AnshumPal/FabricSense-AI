import os
import pandas as pd

# 🔹 CHANGE THIS PATH to your actual dataset folder
folder_path = r"C:\Users\ASUS\OneDrive\Desktop\Textile-dataset\DeepTextile\Aggregated\group_5_10_10"

all_data = []

print("Reading CSV files...")

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        file_path = os.path.join(folder_path, file)
        print(f"Reading: {file}")
        df = pd.read_csv(file_path)
        all_data.append(df)

if len(all_data) == 0:
    print("No CSV files found. Check folder path.")
else:
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df.to_csv("combined_dataset.csv", index=False)
    print("All files combined successfully!")