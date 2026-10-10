import os, json
import numpy as np, pandas as pd
import streamlit as st, altair as alt

GREEN, GOLD = "#3DDC84", "#F2C94C"
st.set_page_config(page_title="YieldQ", page_icon="🌾", layout="wide")

@st.cache_data
def load():
    tr = pd.read_csv("data/train_l.csv"); te = pd.read_csv("data/test_l.csv")
    full = pd.concat([tr, te]).sort_values(["State", "Season", "Crop_Year"]).reset_index(drop=True)
    return full, pd.read_csv("data/forecast_grid.csv"), json.load(open("data/forecast_stats.json"))
full, fc, stats = load()
rmse, err80, naive = stats["rmse"], stats["err80"], stats["naive_rmse"]

with st.sidebar:
    st.markdown("## 🌾 YieldQ")
    st.caption("Farm data input")
    states = sorted(fc["State"].unique())
    state = st.selectbox("State", states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)
    seasons = sorted(fc[fc["State"] == state]["Season"].unique())
    season = st.selectbox("Season", seasons)
    sub = full[(full["State"] == state) & (full["Season"] == season)].sort_values("Crop_Year")
    last_year = int(sub["Crop_Year"].max())
    g = fc[(fc["State"] == state) & (fc["Season"] == season)]
    years = sorted(g["Crop_Year"].unique())
    year = st.selectbox("Forecast year", years, index=len(years) - 1,
                        format_func=lambda y: f"{y} (new forecast)" if y > last_year else f"{y} (backtest)")
    st.selectbox("Intended crop", ["Rice"])
    st.caption("Yield model supports rice. More crops in the comparison step.")
    gy = g[g["Crop_Year"] == year].sort_values("Rain")
    rain0 = float(gy["rain0"].iloc[0])
    rain = st.number_input("Annual rainfall (mm)", min_value=0.0, max_value=8000.0,
                           value=round(rain0, 1), step=10.0, key=f"rain_{state}_{season}_{year}")
    st.markdown("**Soil and climate** (used in the next steps)")
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
    n_ = st.number_input("Nitrogen N (kg/ha)", 0, 400, 90)
    p_ = st.number_input("Phosphorus P (kg/ha)", 0, 200, 40)
    k_ = st.number_input("Potassium K (kg/ha)", 0, 300, 40)
    temp = st.number_input("Average temperature (°C)", 5.0, 45.0, 28.0, 0.5)
    irr = st.selectbox("Irrigation availability", ["None (rainfed)", "Limited", "Adequate"])

is_new = year > last_year
prev = float(gy["Prev"].iloc[0])
actual = gy["Actual"].iloc[0]
pred = float(np.interp(rain, gy["Rain"], gy["Forecast"]))
lo, hi = pred - err80, pred + err80

st.title("YieldQ · Quantum-Hybrid Yield Forecast")
st.caption("Rice, state-level data. Forecast = last season's yield + a change predicted by a classical "
           "encoder and a 4-qubit Qiskit circuit (simulator). Forecasts are computed in advance "
           "for 9 rainfall levels per case; values in between are interpolated.")
if rain < gy["Rain"].min() or rain > gy["Rain"].max():
    st.warning(f"Rainfall is outside the precomputed range ({gy['Rain'].min():.0f} to "
               f"{gy['Rain'].max():.0f} mm). The forecast is held at the nearest edge value.")

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
        if pd.notna(actual):
            st.caption(f"Actual {year} yield: {actual:.2f} t/ha. 'Same as last season' would have been off by "
                       f"{abs(prev - actual):.2f}; the hybrid was off by {abs(pred - actual):.2f}.")
        h = pd.DataFrame({"Year": sub["Crop_Year"].astype(str), "Value": sub["Yield"]})
        f1 = pd.DataFrame({"Year": [str(year)], "Value": [pred], "lo": [lo], "hi": [hi]})
        ys = alt.Scale(zero=False)
        line = alt.Chart(h).mark_line(color=GREEN, point=True).encode(
            x=alt.X("Year:O", title="Year"), y=alt.Y("Value:Q", scale=ys, title="Yield (t/ha)"))
        rule = alt.Chart(f1).mark_rule(color=GOLD, strokeWidth=3).encode(
            x="Year:O", y=alt.Y("lo:Q", scale=ys), y2="hi:Q")
        pt = alt.Chart(f1).mark_point(color=GOLD, size=160, filled=True).encode(
            x="Year:O", y=alt.Y("Value:Q", scale=ys))
        st.altair_chart(line + rule + pt, width="stretch")
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
        import glob
        fs = sorted(glob.glob("results_h4_s*.csv"))
        if fs:
            allr = pd.concat([pd.read_csv(f) for f in fs])
            s = allr.groupby("model")["test_RMSE"].mean().reset_index()
            bars = alt.Chart(s).mark_bar().encode(
                y=alt.Y("model:N", sort="x", title=None),
                x=alt.X("test_RMSE:Q", title="Test RMSE (t/ha, lower is better)"),
                color=alt.condition("indexof(datum.model,'PQC') >= 0", alt.value(GOLD), alt.value(GREEN)))
            st.altair_chart(bars, width="stretch")
            st.caption(f"Mean over {len(fs)} run(s), test years 2017-2019. Gold: Qiskit hybrid.")
with d2:
    with st.container(border=True):
        st.subheader("Quantum circuit")
        st.caption("4 qubits: RY angle encoding, then trainable RealAmplitudes layers. 12 trainable circuit parameters.")
        if os.path.exists("circuit.png"):
            st.image("circuit.png", width="stretch")

st.info(f"Honest note: overall test RMSE is {rmse:.3f} t/ha vs {naive:.3f} for 'same as last season'. "
        "The hybrid is comparable to classical models, not better. State-level data has little "
        "year-to-year signal. Field-level soil, irrigation and satellite data are the next step.")