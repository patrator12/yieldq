import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

NQ = 4
NUM = ["Crop_Year", "Annual_Rainfall", "Log_Fert", "Log_Pest", "Log_Area"]
CAT = ["Season", "State"]

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

class Model(nn.Module):
    def __init__(self, d, head, head_dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, NQ))
        self.head = head
        self.out = nn.Linear(head_dim, 1)
    def forward(self, x):
        z = torch.tanh(self.enc(x)) * np.pi
        return self.out(self.head(z))

_pre = joblib.load("preproc.joblib")
_train = pd.read_csv("data/train.csv")
STATES = sorted(_train["State"].unique())
SEASONS = sorted(_train["Season"].unique())
_RANGE = {"Annual_Rainfall": (_train["Annual_Rainfall"].min(), _train["Annual_Rainfall"].max()),
          "Fert_per_ha": (np.expm1(_train["Log_Fert"].min()), np.expm1(_train["Log_Fert"].max())),
          "Pest_per_ha": (np.expm1(_train["Log_Pest"].min()), np.expm1(_train["Log_Pest"].max()))}
_model = None

def load_model(path="model_pqc_v2.pt"):
    global _model
    if _model is None:
        d = len(_pre["ct"].get_feature_names_out())
        head, hd = quantum_head()
        m = Model(d, head, hd)
        m.load_state_dict(torch.load(path))
        m.eval()
        _model = m
    return _model

def predict(raw):
    """raw: DataFrame with Crop_Year, Season, State, Annual_Rainfall,
    Fert_per_ha, Pest_per_ha, Area. Returns yield in t/ha."""
    d = pd.DataFrame({
        "Crop_Year": raw["Crop_Year"], "Season": raw["Season"], "State": raw["State"],
        "Annual_Rainfall": raw["Annual_Rainfall"],
        "Log_Fert": np.log1p(raw["Fert_per_ha"]),
        "Log_Pest": np.log1p(raw["Pest_per_ha"]),
        "Log_Area": np.log(raw["Area"])})
    X = torch.tensor(_pre["ct"].transform(d[NUM + CAT]), dtype=torch.float32)
    with torch.no_grad():
        p = load_model()(X).numpy().ravel()
    return p * _pre["ys"] + _pre["ym"]

SCENARIOS = [
    ("Fertilizer +10%", "Fert_per_ha", 1.10, True),
    ("Fertilizer +20%", "Fert_per_ha", 1.20, True),
    ("Fertilizer -10%", "Fert_per_ha", 0.90, True),
    ("Pesticide +10%", "Pest_per_ha", 1.10, True),
    ("Pesticide -10%", "Pest_per_ha", 0.90, True),
    ("Rainfall -15% (dry year)", "Annual_Rainfall", 0.85, False),
    ("Rainfall +15% (wet year)", "Annual_Rainfall", 1.15, False),
]

def what_if(field):
    """field: dict with the 7 raw inputs. Returns (base_yield, table, advice)."""
    rows = [dict(field)]
    for _, col, mult, _ in SCENARIOS:
        r = dict(field); r[col] = field[col] * mult; rows.append(r)
    preds = predict(pd.DataFrame(rows))
    base = float(preds[0])
    out = []
    for (label, col, mult, ctrl), p, r in zip(SCENARIOS, preds[1:], rows[1:]):
        lo, hi = _RANGE[col]
        note = "outside training range" if not (lo <= r[col] <= hi) else ""
        pct = (float(p) - base) / base * 100
        if not note and abs(pct) < 3:
            note = "too small to trust"
        out.append({"scenario": label, "yield_t_ha": round(float(p), 2),
                    "change_pct": round(pct, 1), "controllable": ctrl, "note": note})
    table = pd.DataFrame(out)
    ok = table[(table["controllable"]) & (table["note"] == "") & (table["change_pct"] >= 3)]
    if len(ok):
        best = ok.sort_values("change_pct", ascending=False).iloc[0]
        advice = f"Model suggests: {best['scenario']} → about {best['change_pct']:+.1f}% yield."
    else:
        advice = "No clear gain from changing inputs. Keep the current plan."
    return base, table, advice