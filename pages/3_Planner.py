import itertools
import numpy as np, pandas as pd, altair as alt, streamlit as st
from common import CROPS, DEFAULTS, load, crop_yields

GREEN, GOLD = "#3DDC84", "#F2C94C"
st.set_page_config(page_title="YieldQ · Planner", page_icon="🌾", layout="wide")
df, grid = load()

st.title("Production planner")
st.caption("Splits a farm between crops to maximise profit or production within a water supply and a budget.")

c1, c2, c3, c4 = st.columns(4)
states = sorted(df["State"].unique())
state = c1.selectbox("State", states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)
seasons = sorted(df[(df["State"] == state) & df["Crop"].isin(CROPS)]["Season"].unique())
if not seasons:
    st.warning("No data for these crops in this state."); st.stop()
season = c2.selectbox("Season", seasons)
area = c3.number_input("Farm area (ha)", 1.0, 1000.0, 10.0, 1.0)
P = c4.slider("Number of plots", 4, 12, 8)

c5, c6, c7, c8 = st.columns(4)
water_avg = c5.slider("Water available (mm per ha, farm average)", 200, 1500, 700, 25)
budget_ha = c6.slider("Budget (₹ per ha, farm average)", 20000, 200000, 80000, 5000)
share = c7.slider("Max share of land for one crop (%)", 20, 100, 50, 5)
goal = c8.radio("Goal", ["Profit", "Production"], horizontal=True)

yl = crop_yields(df, grid, state, season)
if yl.empty:
    st.warning("No data for this state and season."); st.stop()
st.warning("Water need, cost and price are ILLUSTRATIVE placeholders, not official figures. "
           "Replace them with local data before relying on the plan.")
with st.expander("Assumptions (edit these)"):
    assume = st.data_editor(DEFAULTS, hide_index=True, disabled=["Crop"], key="assume_plan")
t = yl.merge(assume, on="Crop").reset_index(drop=True)
t["Profit (₹/ha)"] = t["Yield (t/ha)"] * 10 * t["Price (₹/quintal)"] - t["Cost (₹/ha)"]

K = len(t)
plot_ha = area / P
cap = max(int(np.floor(share / 100 * P + 1e-9)), int(np.ceil(P / K)))
w, cost = t["Water need (mm)"].values, t["Cost (₹/ha)"].values
prof, yld = t["Profit (₹/ha)"].values, t["Yield (t/ha)"].values

@st.cache_data
def compositions(P, K):
    out = []
    for bars in itertools.combinations(range(P + K - 1), K - 1):
        prev, row = -1, []
        for pos in bars:
            row.append(pos - prev - 1); prev = pos
        row.append(P + K - 1 - prev - 1)
        out.append(row)
    return np.array(out)

def evaluate(C):
    return {"water": C @ (plot_ha * w), "cost": C @ (plot_ha * cost),
            "profit": C @ (plot_ha * prof), "prod": C @ (plot_ha * yld)}

W, B = water_avg * area, budget_ha * area
C = compositions(P, K)
E = evaluate(C)
ok = (E["water"] <= W + 1e-9) & (E["cost"] <= B + 1e-9) & (C.max(axis=1) <= cap)
if not ok.any():
    st.error("No plan fits these limits. Raise the water or budget, or allow a larger share per crop.")
    st.stop()
obj = E["profit"] if goal == "Profit" else E["prod"]
i = int(np.where(ok, obj, -np.inf).argmax())
best = C[i]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total profit (₹)", f"{E['profit'][i]:,.0f}")
m2.metric("Production (t)", f"{E['prod'][i]:,.1f}")
m3.metric("Water used", f"{E['water'][i] / W * 100:.0f}% of available")
m4.metric("Budget used", f"{E['cost'][i] / B * 100:.0f}% of available")

plan = pd.DataFrame({"Crop": t["Crop"], "Plots": best, "Area (ha)": best * plot_ha,
                     "Yield (t/ha)": yld, "Profit (₹)": best * plot_ha * prof})
plan = plan[plan["Plots"] > 0]
d1, d2 = st.columns(2)
with d1:
    with st.container(border=True):
        st.subheader("Recommended land split")
        st.altair_chart(alt.Chart(plan).mark_bar(color=GREEN).encode(
            y=alt.Y("Crop:N", sort="-x", title=None), x=alt.X("Area (ha):Q")), width="stretch")
with d2:
    with st.container(border=True):
        st.subheader("Plan details")
        st.dataframe(plan.round(2), hide_index=True)

def row(name, counts):
    e = evaluate(counts.reshape(1, -1))
    feas = e["water"][0] <= W + 1e-9 and e["cost"][0] <= B + 1e-9
    return {"Plan": name, "Profit (₹)": round(float(e["profit"][0])), "Production (t)": round(float(e["prod"][0]), 1),
            "Water (mm/ha avg)": round(float(e["water"][0] / area)), "Fits limits": "yes" if feas else "no"}

single = None
for j in range(K):
    cj = np.zeros(K, dtype=int); cj[j] = P
    e = evaluate(cj.reshape(1, -1))
    if e["water"][0] <= W + 1e-9 and e["cost"][0] <= B + 1e-9:
        v = e["profit"][0] if goal == "Profit" else e["prod"][0]
        if single is None or v > single[0]:
            single = (v, cj)
eq = np.full(K, P // K); eq[: P % K] += 1
rows = [row("Optimised plan", best)]
rows.append(row("Best single crop that fits", single[1]) if single else
            {"Plan": "Best single crop that fits", "Profit (₹)": "none fits", "Production (t)": "-", "Water (mm/ha avg)": "-", "Fits limits": "-"})
rows.append(row("Equal split across crops", eq))
st.subheader("Compared with simple plans")
st.dataframe(pd.DataFrame(rows), hide_index=True)

st.info(f"How it works: the planner checks all {len(C):,} possible ways to give {P} plots to {K} crops "
        "and keeps the best one that meets the limits, so the answer is exact. For a problem this "
        "small a classical search is instant. Yields are the last recorded yield (rice: Qiskit hybrid "
        "forecast). Results depend on the editable placeholder assumptions and are not advice.")

import json, os
if os.path.exists("data/qaoa_result.json"):
    q = json.load(open("data/qaoa_result.json"))
    with st.expander("Qiskit QAOA experiment: crop selection"):
        st.caption(f"Fixed test case: {q['state']}, {q['season']}, {q['area']:.0f} ha, up to {q['max_crops']} crops, "
                   f"{q['water_avg']:.0f} mm/ha water, ₹{q['budget_ha']:,.0f}/ha budget. "
                   f"{q['qubits']} qubits (one per crop), {q['reps']} QAOA layers, Qiskit statevector simulator.")
        a, b, c = st.columns(3)
        a.metric("Probability of the exact best choice", f"{q['p_exact']*100:.1f}%",
                 f"random guess: {q['random_p_exact']*100:.1f}%", delta_color="off")
        b.metric("Probability of a valid choice", f"{q['p_feasible']*100:.0f}%",
                 f"random guess: {q.get('random_p_feasible', 0)*100:.0f}%", delta_color="off")
        c.metric("Expected profit vs exact optimum", f"{q['expected_profit_ratio']*100:.0f}%",
                 f"random guess: {q.get('random_profit_ratio', 0)*100:.0f}%", delta_color="off")
        st.write(f"Exact best choice (by checking all {q['choices']}): **{q['exact']['crops']}**")
        st.dataframe(pd.DataFrame(q["top"]).rename(columns={"crops": "Choice", "prob": "Probability",
                     "feasible": "Valid", "profit": "Profit (₹)"}), hide_index=True)
        st.caption("This is an experiment. For this size a classical search is exact and instant, so we do "
                   "not claim a quantum advantage. We check the Qiskit circuit against a separate simulation.")