import pandas as pd
import numpy as np

# Load the CSV dataset
df = pd.read_csv('sample_data.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("\nFirst few rows:")
print(df.head())

print("\n" + "=" * 60)
print("BASIC STATISTICS")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)
print(df.describe())

print("\n" + "=" * 60)
print("ADDITIONAL STATISTICS")
print("=" * 60)

# Numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns

for col in numerical_cols:
    print(f"\n{col.upper()}:")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Median: {df[col].median():.2f}")
    print(f"  Std Dev: {df[col].std():.2f}")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Range: {df[col].max() - df[col].min()}")

print("\n" + "=" * 60)
