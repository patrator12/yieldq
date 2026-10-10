import warnings, glob
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch, torch.nn as nn, joblib
import streamlit as st, altair as alt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import matplotlib
matplotlib.use("Agg")

NQ = 4
GREEN, GOLD = "#3DDC84", "#F2C94C"
st.set_page_config(page_title="YieldQ", page_icon="🌾", layout="wide")

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

@st.cache_resource
def circuit_fig():
    return build_circuit()[0].draw("mpl")

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
    full = pd.concat([tr, te]).sort_values(["State", "Season", "Crop_Year"]).reset_index(drop=True)
    full["Momentum"] = full["Prev_Yield"] - full["Roll3_Yield"]
    te = full[full["Crop_Year"] >= 2017].copy()
    head, hd = quantum_head()
    model = Model(len(ct.get_feature_names_out()), head, hd)
    model.load_state_dict(torch.load("model_pqc_s42.pt"))
    model.eval()
    return ct, dm, ds, num, cat, full, te, model

ct, dm, ds, num, cat, full, te, model = load_all()

def forecast(rows):
    df = pd.DataFrame(rows)
    X = torch.tensor(ct.transform(df[num + cat]), dtype=torch.float32)
    with torch.no_grad():
        delta = model(X).numpy().ravel() * ds + dm
    return df["Prev_Yield"].values + delta

@st.cache_data
def test_stats():
    p = forecast(te.to_dict("records"))
    err = np.abs(p - te["Yield"].values)
    return float(np.sqrt(np.mean(err ** 2))), float(np.quantile(err, 0.8))
rmse, err80 = test_stats()

# ---------------- Sidebar: Feature 1, farm data input ----------------
with st.sidebar:
    st.markdown("## 🌾 YieldQ")
    st.caption("Farm data input")
    states = sorted(full["State"].unique())
    state = st.selectbox("State", states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)
    seasons = sorted(full[full["State"] == state]["Season"].unique())
    season = st.selectbox("Season", seasons)
    sub = full[(full["State"] == state) & (full["Season"] == season)].sort_values("Crop_Year")
    last_year = int(sub["Crop_Year"].max())
    years = sorted(set(int(y) for y in sub["Crop_Year"] if y >= 2017) | {last_year + 1})
    year = st.selectbox("Forecast year", years, index=len(years) - 1,
                        format_func=lambda y: f"{y} (new forecast)" if y > last_year else f"{y} (backtest)")
    crop = st.selectbox("Intended crop", ["Rice"])
    st.caption("Yield model supports rice. More crops in the comparison step.")
    is_new = year > last_year
    if not is_new:
        r0 = sub[sub["Crop_Year"] == year].iloc[0]
        rain0 = float(r0["Annual_Rainfall"])
    else:
        rain0 = float(sub["Annual_Rainfall"].iloc[-1])
    rain = st.number_input("Annual rainfall (mm)", min_value=0.0, max_value=8000.0,
                           value=round(rain0, 1), step=10.0, key=f"rain_{state}_{season}_{year}")
    st.markdown("**Soil and climate** (used in the next steps)")
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
    n_ = st.number_input("Nitrogen N (kg/ha)", 0, 400, 90)
    p_ = st.number_input("Phosphorus P (kg/ha)", 0, 200, 40)
    k_ = st.number_input("Potassium K (kg/ha)", 0, 300, 40)
    temp = st.number_input("Average temperature (°C)", 5.0, 45.0, 28.0, 0.5)
    irr = st.selectbox("Irrigation availability", ["None (rainfed)", "Limited", "Adequate"])

# ---------------- Build the model input row ----------------
if is_new:
    prev = float(sub["Yield"].iloc[-1]); roll3 = float(sub["Yield"].iloc[-3:].mean())
    base = sub.iloc[-1]
else:
    prev = float(r0["Prev_Yield"]); roll3 = float(r0["Roll3_Yield"]); base = r0
rain_mean = float((full.loc[full["State"] == state, "Annual_Rainfall"]
                   - full.loc[full["State"] == state, "Rain_Anom"]).iloc[0])
row = {"Crop_Year": year, "Annual_Rainfall": rain, "Rain_Anom": rain - rain_mean,
       "Log_Fert": float(base["Log_Fert"]), "Log_Pest": float(base["Log_Pest"]),
       "Log_Area": float(base["Log_Area"]), "Prev_Yield": prev, "Momentum": prev - roll3,
       "Season": season, "State": state}
pred = float(forecast([row])[0])
lo, hi = pred - err80, pred + err80

# ---------------- Main page ----------------
st.title("YieldQ · Quantum-Hybrid Yield Forecast")
st.caption("Rice, state-level data. Forecast = last season's yield + a change predicted by a classical "
           "encoder and a 4-qubit Qiskit circuit (simulator).")
if rain < 300 or rain > 6550:
    st.warning("Rainfall is outside the range the model was trained on (300 to 6,550 mm). Treat the forecast with care.")

c1, c2 = st.columns([3, 2])
with c1:
    with st.container(border=True):
        st.subheader("Yield prediction")
        a, b, c = st.columns(3)
        a.metric("Last season (t/ha)", f"{prev:.2f}")
        b.metric("Hybrid forecast (t/ha)", f"{pred:.2f}", f"{pred - prev:+.2f} vs last season")
        c.metric("Likely range (t/ha)", f"{lo:.2f} to {hi:.2f}")
        st.caption(f"Range = forecast ± {err80:.2f} t/ha. On the 2017-2019 test years, 80% of forecasts "
                   f"were within this distance of the actual yield.")
        if not is_new:
            st.caption(f"Actual {year} yield: {float(r0['Yield']):.2f} t/ha. 'Same as last season' would have been "
                       f"off by {abs(prev - float(r0['Yield'])):.2f}; the hybrid was off by "
                       f"{abs(pred - float(r0['Yield'])):.2f}.")
        h = pd.DataFrame({"Year": sub["Crop_Year"].astype(str), "Value": sub["Yield"]})
        fc = pd.DataFrame({"Year": [str(year)], "Value": [pred], "lo": [lo], "hi": [hi]})
        ys = alt.Scale(zero=False)
        line = alt.Chart(h).mark_line(color=GREEN, point=True).encode(
            x=alt.X("Year:O", title="Year"), y=alt.Y("Value:Q", scale=ys, title="Yield (t/ha)"))
        rule = alt.Chart(fc).mark_rule(color=GOLD, strokeWidth=3).encode(
            x="Year:O", y=alt.Y("lo:Q", scale=ys), y2="hi:Q")
        pt = alt.Chart(fc).mark_point(color=GOLD, size=160, filled=True).encode(
            x="Year:O", y=alt.Y("Value:Q", scale=ys))
        st.altair_chart(line + rule + pt, width='stretch')
        st.caption("Green: recorded yield. Gold: hybrid forecast with its likely range.")
with c2:
    with st.container(border=True):
        st.subheader("Your farm inputs")
        t1, t2 = st.columns(2)
        t1.metric("Soil pH", f"{ph:.1f}"); t2.metric("Rainfall (mm)", f"{rain:.0f}")
        t1.metric("N / P / K (kg/ha)", f"{n_} / {p_} / {k_}"); t2.metric("Temperature (°C)", f"{temp:.1f}")
        st.metric("Irrigation", irr)
        st.caption("Drives the forecast: state, season, year, rainfall, last 3 recorded yields, "
                   "state farm area, last recorded fertilizer and pesticide. "
                   "Soil pH, N, P, K, temperature and irrigation are used in the crop comparison "
                   "and optimization steps, not by the yield model.")

d1, d2 = st.columns(2)
with d1:
    with st.container(border=True):
        st.subheader("Model comparison")
        fs = sorted(glob.glob("results_h4_s*.csv"))
        if fs:
            allr = pd.concat([pd.read_csv(f) for f in fs])
            s = allr.groupby("model")["test_RMSE"].mean().reset_index()
            bars = alt.Chart(s).mark_bar().encode(
                y=alt.Y("model:N", sort="x", title=None), x=alt.X("test_RMSE:Q", title="Test RMSE (t/ha, lower is better)"),
                color=alt.condition("indexof(datum.model,'PQC') >= 0", alt.value(GOLD), alt.value(GREEN)))
            st.altair_chart(bars, width='stretch')
            st.caption(f"Mean over {len(fs)} run(s), test years 2017-2019. Gold: Qiskit hybrid.")
with d2:
    with st.container(border=True):
        st.subheader("Quantum circuit")
        st.caption("4 qubits: RY angle encoding, then trainable RealAmplitudes layers. 12 trainable circuit parameters.")
        ##st.pyplot(circuit_fig())

st.info(f"Honest note: overall test RMSE is {rmse:.3f} t/ha, comparable to classical models and "
        "close to the 'same as last season' rule. State-level data has little year-to-year signal. "
        "Field-level soil, irrigation and satellite data are the next step.")