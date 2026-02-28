# 🚀 STEP 1 — Basic EDA (Exploratory Data Analysis)
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("combined_dataset.csv")

print("🔹 First 5 rows:")
print(df.head())

print("\n🔹 Shape of dataset:")
print(df.shape)

print("\n🔹 Columns:")
print(df.columns)

print("\n🔹 Data Types:")
print(df.dtypes)

print("\n🔹 Missing Values:")
print(df.isnull().sum())

print("\n🔹 Duplicate Rows:")
print(df.duplicated().sum())

print("\n🔹 Statistical Summary:")
print(df.describe())


# 🚀 STEP 2 — Data Cleaning
# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

print("\nAfter Cleaning Shape:", df.shape)