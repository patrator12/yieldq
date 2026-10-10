import numpy as np, pandas as pd
from yieldq_core import predict, what_if

test = pd.read_csv("data/test.csv")
raw = pd.DataFrame({
    "Crop_Year": test["Crop_Year"], "Season": test["Season"], "State": test["State"],
    "Annual_Rainfall": test["Annual_Rainfall"],
    "Fert_per_ha": np.expm1(test["Log_Fert"]),
    "Pest_per_ha": np.expm1(test["Log_Pest"]),
    "Area": np.exp(test["Log_Area"])})

p = predict(raw)
rmse = float(np.sqrt(np.mean((p - test["Yield"].values) ** 2)))
print(f"RMSE on test via predict(): {rmse:.3f}  (training run printed 0.399)")

field = raw.iloc[0].to_dict()
print("\nExample field:", field)
print("Actual yield:", round(float(test["Yield"].iloc[0]), 2))
base, table, advice = what_if(field)
print("Predicted yield:", round(base, 2), "t/ha")
print(table.to_string(index=False))
print(advice)