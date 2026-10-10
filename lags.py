import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

tr = pd.read_csv("data/train.csv"); te = pd.read_csv("data/test.csv")
df = pd.concat([tr, te]).sort_values(["State", "Season", "Crop_Year"]).reset_index(drop=True)

g = df.groupby(["State", "Season"])["Yield"]
df["Prev_Yield"] = g.shift(1)
df["Roll3_Yield"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
rain_mean = df[df["Crop_Year"] <= 2016].groupby("State")["Annual_Rainfall"].mean()
df["Rain_Anom"] = df["Annual_Rainfall"] - df["State"].map(rain_mean).fillna(df["Annual_Rainfall"].mean())

n0 = len(df)
df = df.dropna(subset=["Prev_Yield"]).reset_index(drop=True)
print(f"rows: {n0} -> {len(df)} (dropped first appearance of each state-season)")

train = df[df["Crop_Year"] <= 2016].reset_index(drop=True)
test = df[df["Crop_Year"] >= 2017].reset_index(drop=True)
train.to_csv("data/train_l.csv", index=False); test.to_csv("data/test_l.csv", index=False)
print("train:", len(train), "test:", len(test))

num = ["Crop_Year", "Annual_Rainfall", "Log_Fert", "Log_Pest", "Log_Area",
       "Prev_Yield", "Roll3_Yield", "Rain_Anom"]
cat = ["Season", "State"]
ytest = test["Yield"].values
rmse = lambda p: float(np.sqrt(mean_squared_error(ytest, p)))

rows = [{"model": "Naive: last year's yield", "RMSE": rmse(test["Prev_Yield"]),
         "R2": r2_score(ytest, test["Prev_Yield"])},
        {"model": "Naive: avg of last 3", "RMSE": rmse(test["Roll3_Yield"]),
         "R2": r2_score(ytest, test["Roll3_Yield"])}]

models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                        early_stopping=True, random_state=42)}
for name, m in models.items():
    pipe = Pipeline([("prep", ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])), ("model", m)])
    pipe.fit(train[num + cat], train["Yield"])
    p = pipe.predict(test[num + cat])
    rows.append({"model": name, "RMSE": rmse(p), "R2": r2_score(ytest, p)})

print("\n", pd.DataFrame(rows).sort_values("RMSE").round(3).to_string(index=False))