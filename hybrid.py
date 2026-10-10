import sys, time, warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

torch.manual_seed(42); np.random.seed(42)
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
BATCH, LR, NQ = 64, 0.02, 4

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
num = ["Crop_Year", "Annual_Rainfall", "Log_Fert", "Log_Pest", "Log_Area"]
cat = ["Season", "State"]
ct = ColumnTransformer([("num", StandardScaler(), num),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                       sparse_threshold=0)
Xtr = torch.tensor(ct.fit_transform(train[num + cat]), dtype=torch.float32)
Xte = torch.tensor(ct.transform(test[num + cat]), dtype=torch.float32)
ym, ys = train["Yield"].mean(), train["Yield"].std()
ytr = torch.tensor(((train["Yield"] - ym) / ys).values, dtype=torch.float32).unsqueeze(1)
yte = test["Yield"].values
import joblib
joblib.dump({"ct": ct, "ym": float(ym), "ys": float(ys)}, "preproc.joblib")
print("input features:", Xtr.shape[1], "| epochs:", EPOCHS)

def quantum_head():
    fm = ZZFeatureMap(NQ, reps=1)
    an = RealAmplitudes(NQ, reps=2)
    qc = QuantumCircuit(NQ)
    qc.compose(fm, inplace=True); qc.compose(an, inplace=True)
    qnn = EstimatorQNN(circuit=qc, input_params=fm.parameters,
                       weight_params=an.parameters, input_gradients=True)
    return TorchConnector(qnn)

def classical_head():
    return nn.Sequential(nn.Linear(NQ, 3), nn.Tanh(), nn.Linear(3, 1))

class Model(nn.Module):
    def __init__(self, d, head):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, NQ))
        self.head = head
        self.out = nn.Linear(1, 1)
    def forward(self, x):
        z = torch.tanh(self.enc(x)) * np.pi
        return self.out(self.head(z))

def evaluate(model):
    model.eval()
    with torch.no_grad():
        p = model(Xte).numpy().ravel() * ys + ym
    return (np.sqrt(mean_squared_error(yte, p)),
            mean_absolute_error(yte, p), r2_score(yte, p))

def run(name, head):
    model = Model(Xtr.shape[1], head)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    head_par = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"\n=== {name} | trainable params: {n_par} (head: {head_par}) ===")
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        t0 = time.time(); model.train()
        perm = torch.randperm(len(Xtr)); tot = 0.0
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        if ep == 1 or ep % 5 == 0 or ep == EPOCHS:
            rmse, _, _ = evaluate(model)
            print(f"epoch {ep:3d} | train loss {tot/len(Xtr):.4f} | "
                  f"test RMSE {rmse:.3f} | {time.time()-t0:.1f}s/epoch")
    rmse, mae, r2 = evaluate(model)
    tag = "pqc" if "PQC" in name else "classical"
    torch.save(model.state_dict(), f"model_{tag}.pt")
    return {"model": name, "params": n_par, "RMSE": rmse, "MAE": mae, "R2": r2,
            "train_sec": time.time() - t_start}

rows = [run("Encoder + classical head", classical_head()),
        run("Encoder + Qiskit PQC", quantum_head())]
res = pd.DataFrame(rows).round(3)
print("\n", res.to_string(index=False))
res.to_csv("results_hybrid.csv", index=False)