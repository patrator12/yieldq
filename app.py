import warnings, glob
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
import streamlit as st
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

NQ = 4
st.set_page_config(page_title="YieldQ", layout="wide")

def build_circuit():
    x = ParameterVector("x", NQ)
    an = RealAmplitudes(NQ, reps=2)
    qc = QuantumCircuit(NQ)
    for i in range(NQ):
        qc.ry(x[i], i)
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
        self.head = head
        self.out = nn.Linear(hd, 1)
    def forward(self, x):
        return self.out(self.head(torch.tanh(self.enc(x)) * np.pi))

@st.cache_resource
def load_all():
    pre = joblib.load("preproc_l.joblib")
    ct, dm, ds, num, cat = pre["ct"], pre["dm"], pre["ds"], pre["num"], pre["cat"]
    tr = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
    for d in (tr, te):
        d["Momentum"] = d["Prev_Yield"] - d["Roll3_Yield"]
    X = torch.tensor(ct.transform(te[num + cat]), dtype=torch.float32)
    head, hd = quantum_head()
    model = Model(X.shape[1], head, hd)
    model.load_state_dict(torch.load("model_pqc_s42.pt"))
    model.eval()
    with torch.no_grad():
        delta = model(X).numpy().ravel() * ds + dm
    te = te.copy()
    te["Forecast"] = te["Prev_Yield"] + delta
    return tr, te

st.title("YieldQ: Quantum-Hybrid Rice Yield Forecast")
st.caption("One-step-ahead forecast: last season's yield + a change predicted by a classical "
           "encoder and a 4-qubit Qiskit circuit. State-level data (Kaggle India crop yield, rice).")

with st.spinner("Loading model and computing forecasts..."):
    tr, te = load_all()
hist = pd.concat([tr, te])
rmse = float(np.sqrt(((te["Forecast"] - te["Yield"]) ** 2).mean()))
naive = float(np.sqrt(((te["Prev_Yield"] - te["Yield"]) ** 2).mean()))
st.write(f"Held-out test years 2017-2019 ({len(te)} rows): "
         f"**Hybrid test RMSE {rmse:.3f}** vs naive 'same as last year' {naive:.3f} t/ha")

c1, c2, c3 = st.columns(3)
state = c1.selectbox("State", sorted(te["State"].unique()))
seasons = sorted(te[te["State"] == state]["Season"].unique())
season = c2.selectbox("Season", seasons)
sub = te[(te["State"] == state) & (te["Season"] == season)]
year = c3.selectbox("Held-out year", sorted(sub["Crop_Year"].unique()))
r = sub[sub["Crop_Year"] == year].iloc[0]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Last season's yield (t/ha)", f"{r['Prev_Yield']:.2f}")
m2.metric("Hybrid forecast (t/ha)", f"{r['Forecast']:.2f}",
          f"{r['Forecast'] - r['Prev_Yield']:+.2f} vs last season")
m3.metric("Actual yield (t/ha)", f"{r['Yield']:.2f}")
m4.metric("Forecast error (t/ha)", f"{abs(r['Forecast'] - r['Yield']):.2f}")
st.caption(f"For comparison, 'same as last season' would have been off by "
           f"{abs(r['Prev_Yield'] - r['Yield']):.2f} t/ha.")

h = hist[(hist["State"] == state) & (hist["Season"] == season)].sort_values("Crop_Year")
chart = pd.DataFrame({"Actual yield": h.set_index("Crop_Year")["Yield"]})
chart["Hybrid forecast (held-out years)"] = sub.set_index("Crop_Year")["Forecast"]
chart.index = chart.index.astype(str)
st.line_chart(chart)

st.subheader("Model comparison (from our runs)")
fs = sorted(glob.glob("results_h4_s*.csv"))
if fs:
    allr = pd.concat([pd.read_csv(f) for f in fs])
    summ = allr.groupby("model")[["val_RMSE", "test_RMSE", "test_R2"]].mean().round(3)
    st.dataframe(summ.sort_values("test_RMSE"))
    st.caption(f"Mean over {len(fs)} seed run(s). Lower RMSE is better.")

with st.expander("How the quantum part works"):
    st.write("A classical encoder compresses 44 input features into 4 angles. "
             "Each angle sets one qubit (RY gate). Trainable RealAmplitudes layers "
             "entangle the qubits. We read each qubit's Z value (4 numbers) and a "
             "linear layer turns them into the predicted change in yield. "
             "The quantum head has 12 trainable parameters.")
    st.pyplot(build_circuit()[0].draw("mpl"))

st.info("Limitations: state-level data with little year-to-year signal, so a simple "
        "'same as last year' rule is already strong. The hybrid model is comparable to "
        "classical models, not better. Running on a simulator, not quantum hardware. "
        "Field-level soil, irrigation and satellite data are the next step.")