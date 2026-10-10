import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor

df = pd.read_csv("data/rice_clean.csv")
n0 = len(df)
df = df[df["Crop_Year"] <= 2019]
df = df[(df["Yield"] >= 0.3) & (df["Yield"] <= 6)].reset_index(drop=True)
print(f"rows: {n0} -> {len(df)} (dropped 2020 and yield outliers)")

# log-transform skewed inputs; the app must apply the same log1p to its inputs
df["Log_Fert"] = np.log1p(df["Fert_per_ha"])
df["Log_Pest"] = np.log1p(df["Pest_per_ha"])
df = df.drop(columns=["Fert_per_ha", "Pest_per_ha"])

train = df[df["Crop_Year"] <= 2016].reset_index(drop=True)
test = df[df["Crop_Year"] >= 2017].reset_index(drop=True)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("train:", len(train), "test:", len(test))

num = ["Crop_Year", "Annual_Rainfall", "Log_Fert", "Log_Pest", "Log_Area"]
cat = ["Season", "State"]
X_tr, y_tr = train[num + cat], train["Yield"]
X_te, y_te = test[num + cat], test["Yield"]

def prep():
    return ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])

models = {
    "Mean (dummy)": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                        early_stopping=True, random_state=42),
}

rows = []
for name, m in models.items():
    pipe = Pipeline([("prep", prep()), ("model", m)])
    pipe.fit(X_tr, y_tr)
    p = pipe.predict(X_te)
    rows.append({"model": name,
                 "RMSE": np.sqrt(mean_squared_error(y_te, p)),
                 "MAE": mean_absolute_error(y_te, p),
                 "R2": r2_score(y_te, p)})

res = pd.DataFrame(rows).sort_values("RMSE").round(3)
print("\n", res.to_string(index=False))
res.to_csv("results_baseline.csv", index=False)