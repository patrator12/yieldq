import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

tr = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
for d in (tr, te):
    d["Momentum"] = d["Prev_Yield"] - d["Roll3_Yield"]
    d["Delta"] = d["Yield"] - d["Prev_Yield"]

num = ["Crop_Year", "Annual_Rainfall", "Rain_Anom", "Log_Fert", "Log_Pest",
       "Log_Area", "Prev_Yield", "Momentum"]
cat = ["Season", "State"]
y = te["Yield"].values
rmse = lambda p: float(np.sqrt(mean_squared_error(y, p)))
rows = [{"model": "Naive: last year", "RMSE": rmse(te["Prev_Yield"]),
         "R2": r2_score(y, te["Prev_Yield"])}]

models = {
    "Ridge a=10": Ridge(alpha=10),
    "Ridge a=100": Ridge(alpha=100),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                            subsample=0.8, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, max_iter=3000,
                        early_stopping=True, random_state=42)}
for name, m in models.items():
    pipe = Pipeline([("prep", ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])), ("model", m)])
    pipe.fit(tr[num + cat], tr["Delta"])
    p = te["Prev_Yield"].values + pipe.predict(te[num + cat])
    rows.append({"model": name + " (predicts change)", "RMSE": rmse(p), "R2": r2_score(y, p)})

print(pd.DataFrame(rows).sort_values("RMSE").round(3).to_string(index=False))