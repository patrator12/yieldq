import sys, time, copy, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
torch.manual_seed(SEED); np.random.seed(SEED)
BATCH, LR, NQ = 64, 0.02, 4

full = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
for d in (full, te):
    d["Momentum"] = d["Prev_Yield"] - d["Roll3_Yield"]
    d["Delta"] = d["Yield"] - d["Prev_Yield"]
tr = full[full["Crop_Year"] <= 2014].reset_index(drop=True)
va = full[full["Crop_Year"] >= 2015].reset_index(drop=True)

num = ["Crop_Year", "Annual_Rainfall", "Rain_Anom", "Log_Fert", "Log_Pest",
       "Log_Area", "Prev_Yield", "Momentum"]
cat = ["Season", "State"]
ct = ColumnTransformer([("num", StandardScaler(), num),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                       sparse_threshold=0)
Xtr_np = ct.fit_transform(tr[num + cat])
Xva_np = ct.transform(va[num + cat]); Xte_np = ct.transform(te[num + cat])
T = lambda a: torch.tensor(a, dtype=torch.float32)
Xtr, Xva, Xte = T(Xtr_np), T(Xva_np), T(Xte_np)
dm, ds = float(tr["Delta"].mean()), float(tr["Delta"].std())
ytr = T(((tr["Delta"] - dm) / ds).values).unsqueeze(1)
joblib.dump({"ct": ct, "dm": dm, "ds": ds, "num": num, "cat": cat}, "preproc_l.joblib")
print(f"train {len(tr)} | val {len(va)} | test {len(te)} | features {Xtr.shape[1]} | epochs {EPOCHS} | seed {SEED}")

rm = lambda y, p: float(np.sqrt(mean_squared_error(y, p)))
def row(name, p, extra=None):
    r = {"model": name, "val_RMSE": rm(va["Yield"], p[0]), "test_RMSE": rm(te["Yield"], p[1]),
         "test_MAE": mean_absolute_error(te["Yield"], p[1]), "test_R2": r2_score(te["Yield"], p[1])}
    r.update(extra or {}); return r

def quantum_head():
    x = ParameterVector("x", NQ)
    an = RealAmplitudes(NQ, reps=2)
    qc = QuantumCircuit(NQ)
    for i in range(NQ): qc.ry(x[i], i)
    qc.compose(an, inplace=True)
    obs = [SparsePauliOp("I" * (NQ - 1 - i) + "Z" + "I" * i) for i in range(NQ)]
    qnn = EstimatorQNN(circuit=qc, observables=obs, input_params=list(x),
                       weight_params=an.parameters, input_gradients=True)
    return TorchConnector(qnn), NQ

def classical_head():
    return nn.Sequential(nn.Linear(NQ, 3), nn.Tanh(), nn.Linear(3, 1)), 1

class Model(nn.Module):
    def __init__(self, d, head, hd):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, NQ))
        self.head = head; self.out = nn.Linear(hd, 1)
    def forward(self, x):
        return self.out(self.head(torch.tanh(self.enc(x)) * np.pi))

def pred(model, X, prev):
    model.eval()
    with torch.no_grad():
        return prev + model(X).numpy().ravel() * ds + dm

pv, pt = va["Prev_Yield"].values, te["Prev_Yield"].values

def run(name, hd, tag):
    head, hdim = hd
    model = Model(Xtr.shape[1], head, hdim)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n=== {name} | params {n_par} ===")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    lossf = nn.MSELoss(); best, best_ep, best_state = 1e9, 0, None; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]; opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx]); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        v = rm(va["Yield"], pred(model, Xva, pv))
        if v < best: best, best_ep, best_state = v, ep, copy.deepcopy(model.state_dict())
        print(f"epoch {ep:2d} | train loss {tot/len(Xtr):.3f} | val RMSE {v:.3f}")
    model.load_state_dict(best_state)
    torch.save(best_state, f"model_{tag}_s{SEED}.pt")
    print(f"best epoch by validation: {best_ep}")
    return row(name, (pred(model, Xva, pv), pred(model, Xte, pt)),
               {"params": n_par, "best_ep": best_ep, "train_sec": round(time.time() - t0)})

rows = [row("Naive: last year", (pv, pt))]
ridge = Ridge(alpha=10).fit(Xtr_np, tr["Delta"])
rows.append(row("Ridge (predicts change)",
                (pv + ridge.predict(Xva_np), pt + ridge.predict(Xte_np))))
rows.append(run("Encoder + classical head", classical_head(), "cls"))
rows.append(run("Encoder + Qiskit PQC", quantum_head(), "pqc"))
res = pd.DataFrame(rows).round(3)
print("\n", res.to_string(index=False))
res.to_csv(f"results_h4_s{SEED}.csv", index=False)