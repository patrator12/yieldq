import warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import json, numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

NQ = 4
def build_circuit():
    x = ParameterVector("x", NQ); an = RealAmplitudes(NQ, reps=2)
    qc = QuantumCircuit(NQ)
    for i in range(NQ): qc.ry(x[i], i)
    qc.compose(an, inplace=True)
    return qc, x, an
def quantum_head():
    qc, x, an = build_circuit()
    obs = [SparsePauliOp("I" * (NQ - 1 - i) + "Z" + "I" * i) for i in range(NQ)]
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=list(x),
                       weight_params=an.parameters, input_gradients=True)
    return TorchConnector(qnn), NQ
class Model(nn.Module):
    def __init__(self, d, head, hd):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, NQ))
        self.head = head; self.out = nn.Linear(hd, 1)
    def forward(self, x):
        return self.out(self.head(torch.tanh(self.enc(x)) * np.pi))

pre = joblib.load("preproc_l.joblib")
ct, dm, ds, num, cat = pre["ct"], pre["dm"], pre["ds"], pre["num"], pre["cat"]
tr = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
full = pd.concat([tr, te]).sort_values(["State", "Season", "Crop_Year"]).reset_index(drop=True)
full["Momentum"] = full["Prev_Yield"] - full["Roll3_Yield"]
head, hd = quantum_head()
model = Model(len(ct.get_feature_names_out()), head, hd)
model.load_state_dict(torch.load("model_pqc_s42.pt")); model.eval()

def forecast(df):
    out = []
    for i in range(0, len(df), 300):
        X = torch.tensor(ct.transform(df.iloc[i:i + 300][num + cat]), dtype=torch.float32)
        with torch.no_grad():
            out.append(model(X).numpy().ravel() * ds + dm)
        print(f"  {min(i + 300, len(df))}/{len(df)}", flush=True)
    return df["Prev_Yield"].values + np.concatenate(out)

# test statistics (2017-2019)
t = full[full["Crop_Year"] >= 2017].copy()
p = forecast(t); err = np.abs(p - t["Yield"].values)
stats = {"rmse": float(np.sqrt(np.mean(err ** 2))), "err80": float(np.quantile(err, 0.8)),
         "naive_rmse": float(np.sqrt(np.mean((t["Prev_Yield"] - t["Yield"]) ** 2)))}
json.dump(stats, open("data/forecast_stats.json", "w")); print(stats)

# forecast grid
MULTS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
rows = []
for state in full["State"].unique():
    rain_mean = float((full.loc[full["State"] == state, "Annual_Rainfall"]
                       - full.loc[full["State"] == state, "Rain_Anom"]).iloc[0])
    for season in full[full["State"] == state]["Season"].unique():
        sub = full[(full["State"] == state) & (full["Season"] == season)].sort_values("Crop_Year")
        last_year = int(sub["Crop_Year"].max())
        years = [int(y) for y in sub["Crop_Year"] if y >= 2017] + [last_year + 1]
        for year in years:
            if year <= last_year:
                r = sub[sub["Crop_Year"] == year].iloc[0]
                prev, roll3, rain0, base, actual = r["Prev_Yield"], r["Roll3_Yield"], r["Annual_Rainfall"], r, r["Yield"]
            else:
                prev, roll3 = sub["Yield"].iloc[-1], sub["Yield"].iloc[-3:].mean()
                rain0, base, actual = sub["Annual_Rainfall"].iloc[-1], sub.iloc[-1], np.nan
            for m in MULTS:
                rain = float(rain0) * m
                rows.append({"State": state, "Season": season, "Crop_Year": year, "Rain": rain,
                             "rain0": float(rain0), "Prev": float(prev), "Actual": actual,
                             "Annual_Rainfall": rain, "Rain_Anom": rain - rain_mean,
                             "Log_Fert": float(base["Log_Fert"]), "Log_Pest": float(base["Log_Pest"]),
                             "Log_Area": float(base["Log_Area"]), "Prev_Yield": float(prev),
                             "Momentum": float(prev - roll3)})
g = pd.DataFrame(rows)
print("grid rows:", len(g))
g["Forecast"] = forecast(g)
g[["State", "Season", "Crop_Year", "Rain", "rain0", "Prev", "Actual", "Forecast"]].to_csv(
    "data/forecast_grid.csv", index=False)
print("saved data/forecast_grid.csv")

# circuit picture last, so the forecasts are already saved
try:
    fig = build_circuit()[0].draw("mpl")
    fig.savefig("circuit.png", dpi=150, bbox_inches="tight"); print("saved circuit.png")
except Exception as e:
    print("circuit drawing failed:", e)