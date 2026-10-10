import pandas as pd

df = pd.read_csv("data/crop_yield.csv")
print("shape:", df.shape)
print("columns:", df.columns.tolist())
print(df.head())
print("\nmissing values:\n", df.isna().sum())
print("\ntop crops:\n", df["Crop"].value_counts().head(10))
print("\nyears:", df["Crop_Year"].min(), "to", df["Crop_Year"].max())
print("\nstates:", df["State"].nunique())