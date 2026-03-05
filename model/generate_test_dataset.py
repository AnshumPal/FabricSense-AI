import pandas as pd

# Load the combined dataset
df = pd.read_csv("combined_dataset.csv")

# Keep ONLY spectral bands (C000–C223)
bands = [f'C{i:03d}' for i in range(224)]
df_bands = df[bands]

# Randomly sample rows for testing
test_df = df_bands.sample(n=50)

# Save test dataset
test_df.to_csv("test_dataset.csv", index=False)

print("Test dataset generated successfully!")
print("Shape:", test_df.shape)