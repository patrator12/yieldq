import numpy as np, pandas as pd
from yieldq_core import predict

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
recent = train[train["Crop_Year"] >= 2012]
b_ss = recent.groupby(["State", "Season"])["Yield"].mean()
b_s = recent.groupby("State")["Yield"].mean()
overall = recent["Yield"].mean()

def baseline(state, season):
    if (state, season) in b_ss.index: return b_ss[(state, season)]
    if state in b_s.index: return b_s[state]
    return overall

raw = pd.DataFrame({
    "Crop_Year": test["Crop_Year"], "Season": test["Season"], "State": test["State"],
    "Annual_Rainfall": test["Annual_Rainfall"],
    "Fert_per_ha": np.expm1(test["Log_Fert"]), "Pest_per_ha": np.expm1(test["Log_Pest"]),
    "Area": np.exp(test["Log_Area"])})
test["pred"] = predict(raw)
test["base"] = [baseline(s, se) for s, se in zip(test["State"], test["Season"])]

rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
print(f"RMSE model:                      {rmse(test['pred'], test['Yield']):.3f}")
print(f"RMSE 'recent average' baseline:  {rmse(test['base'], test['Yield']):.3f}")

T = 0.4
a = test["Yield"] < test["base"] - T
p = test["pred"] < test["base"] - T
tp = int((a & p).sum()); fp = int((~a & p).sum()); fn = int((a & ~p).sum())
print(f"\nRows: {len(test)} | really below normal: {int(a.sum())} | flagged: {int(p.sum())}")
print(f"Caught (true positives): {tp} | false alarms: {fp} | missed: {fn}")
if p.sum(): print(f"Precision: {tp/p.sum():.2f}")
if a.sum(): print(f"Recall:    {tp/a.sum():.2f}")