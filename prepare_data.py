import pandas as pd
import numpy as np

df = pd.read_csv("data/crop_yield.csv")

# remove stray spaces in text columns
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].str.strip()

df = df[df["Crop"] == "Rice"].copy()

# per-hectare features
df = df[(df["Area"] > 0) & (df["Yield"] > 0)]
df["Fert_per_ha"] = df["Fertilizer"] / df["Area"]
df["Pest_per_ha"] = df["Pesticide"] / df["Area"]
df["Log_Area"] = np.log(df["Area"])

keep = ["Crop_Year", "Season", "State", "Annual_Rainfall",
        "Fert_per_ha", "Pest_per_ha", "Log_Area", "Yield"]
df = df[keep].reset_index(drop=True)

print("rows:", len(df))
print("\nYield summary:\n", df["Yield"].describe())
print("\nrows per year:\n", df["Crop_Year"].value_counts().sort_index())
print("\nseasons:\n", df["Season"].value_counts())
print("\nstates:", df["State"].nunique())
print("\ncorrelation with Yield:\n",
      df.drop(columns=["Season", "State"]).corr()["Yield"].round(2))

df.to_csv("data/rice_clean.csv", index=False)
print("\nsaved data/rice_clean.csv")