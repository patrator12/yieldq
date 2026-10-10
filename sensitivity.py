import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from yieldq_core import predict, _pre, NUM, CAT, SCENARIOS, _RANGE

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")

def to_raw(d):
    return pd.DataFrame({"Crop_Year": d["Crop_Year"], "Season": d["Season"],
        "State": d["State"], "Annual_Rainfall": d["Annual_Rainfall"],
        "Fert_per_ha": np.expm1(d["Log_Fert"]), "Pest_per_ha": np.expm1(d["Log_Pest"]),
        "Area": np.exp(d["Log_Area"])})

def to_model_df(r):
    return pd.DataFrame({"Crop_Year": r["Crop_Year"], "Season": r["Season"],
        "State": r["State"], "Annual_Rainfall": r["Annual_Rainfall"],
        "Log_Fert": np.log1p(r["Fert_per_ha"]), "Log_Pest": np.log1p(r["Pest_per_ha"]),
        "Log_Area": np.log(r["Area"])})

raw = to_raw(test)
ridge = Ridge(alpha=1.0).fit(_pre["ct"].transform(train[NUM + CAT]), train["Yield"])
ridge_pred = lambda r: ridge.predict(_pre["ct"].transform(to_model_df(r)[NUM + CAT]))

base_q, base_r = predict(raw), ridge_pred(raw)
print("Training ranges:", {k: (round(a, 2), round(b, 2)) for k, (a, b) in _RANGE.items()})
print()
for label, col, mult, _ in SCENARIOS:
    r2 = raw.copy(); r2[col] = r2[col] * mult
    lo, hi = _RANGE[col]
    inr = ((r2[col] >= lo) & (r2[col] <= hi)).mean() * 100
    dq = ((predict(r2) - base_q) / base_q * 100).mean()
    dr = ((ridge_pred(r2) - base_r) / base_r * 100).mean()
    print(f"{label:26s} PQC {dq:+5.1f}% | Ridge {dr:+5.1f}% | in range {inr:3.0f}%")