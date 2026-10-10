import sys, time, warnings
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

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
torch.manual_seed(SEED); np.random.seed(SEED)
BATCH, LR, NQ = 64, 0.02, 4

tr = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
for d in (tr, te):
    d["Momentum"] = d["Prev_Yield"] - d["Roll3_Yield"]
    d["Delta"] = d["Yield"] - d["Prev_Yield"]
num = ["Crop_Year", "Annual_Rainfall", "Rain_Anom", "Log_Fert", "Log_Pest",
       "Log_Area", "Prev_Yield", "Momentum"]
cat = ["Season", "State"]
ct = ColumnTransformer([("num", StandardScaler(), num),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                       sparse_threshold=0)
Xtr_np = ct.fit_transform(tr[num + cat]); Xte_np = ct.transform(te[num + cat])
Xtr = torch.tensor(Xtr_np, dtype=torch.float32); Xte = torch.tensor(Xte_np, dtype=torch.float32)
dm, ds = float(tr["Delta"].mean()), float(tr["Delta"].std())
ytr = torch.tensor(((tr["Delta"] - dm) / ds).values, dtype=torch.float32).unsqueeze(1)
prev_te, yte = te["Prev_Yield"].values, te["Yield"].values
joblib.dump({"ct": ct, "dm": dm, "ds": ds, "num": num, "cat": cat}, "preproc_l.joblib")
print("input features:", Xtr.shape[1], "| epochs:", EPOCHS, "| seed:", SEED)

def metrics(name, p, n_par=None, secs=None):
    return {"model": name, "params": n_par, "RMSE": float(np.sqrt(mean_squared_error(yte, p))),
            "MAE": mean_absolute_error(yte, p), "R2": r2_score(yte, p), "train_sec": secs}

def quantum_head():
    x = ParameterVector("x", NQ)
    an = RealAmplitudes(NQ, reps=2)
    qc = QuantumCircuit(NQ)
    for i in range(NQ):
        qc.ry(x[i], i)
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
        self.head = head
        self.out = nn.Linear(hd, 1)
    def forward(self, x):
        z = torch.tanh(self.enc(x)) * np.pi
        return self.out(self.head(z))

def predict_yield(model):
    model.eval()
    with torch.no_grad():
        delta = model(Xte).numpy().ravel() * ds + dm
    return prev_te + delta

def run(name, head_and_dim, tag):
    head, hd = head_and_dim
    model = Model(Xtr.shape[1], head, hd)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hp = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== {name} | trainable params: {n_par} (head: {hp}) ===")
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    lossf = nn.MSELoss(); t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train()
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            r = np.sqrt(mean_squared_error(yte, predict_yield(model)))
            print(f"epoch {ep:3d} | train loss {tot/len(Xtr):.4f} | test RMSE {r:.3f} | {time.time()-t0:.1f}s/epoch")
    torch.save(model.state_dict(), f"model_{tag}_s{SEED}.pt")
    return metrics(name, predict_yield(model), n_par, time.time() - t_start)

rows = [metrics("Naive: last year", prev_te)]
ridge = Ridge(alpha=10).fit(Xtr_np, tr["Delta"])
rows.append(metrics("Ridge (predicts change)", prev_te + ridge.predict(Xte_np)))
rows.append(run("Encoder + classical head", classical_head(), "cls_l"))
rows.append(run("Encoder + Qiskit PQC", quantum_head(), "pqc_l"))
res = pd.DataFrame(rows).round(3)
print("\n", res.to_string(index=False))
res.to_csv(f"results_l_s{SEED}.csv", index=False)