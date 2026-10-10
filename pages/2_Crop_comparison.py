import numpy as np, pandas as pd, altair as alt, streamlit as st

GREEN, GOLD = "#3DDC84", "#F2C94C"
st.set_page_config(page_title="YieldQ · Crop comparison", page_icon="🌾", layout="wide")

CROPS = ["Rice", "Maize", "Groundnut", "Dry chillies", "Jowar", "Gram", "Sunflower"]
DEFAULTS = pd.DataFrame({
    "Crop": CROPS,
    "Water need (mm)": [1200, 600, 550, 800, 450, 350, 500],
    "Cost (₹/ha)": [60000, 50000, 55000, 150000, 30000, 30000, 35000],
    "Price (₹/quintal)": [2300, 2200, 6000, 15000, 3300, 5500, 6500]})

@st.cache_data
def load():
    df = pd.read_csv("data/crop_yield.csv")
    for c in ["Crop", "Season", "State"]:
        df[c] = df[c].str.strip()
    return df, pd.read_csv("data/forecast_grid.csv")
df, grid = load()

st.title("Crop comparison")
st.caption("Compares crops for one state and season using expected yield, water need, cost and profit.")

a, b, c = st.columns(3)
states = sorted(df["State"].unique())
state = a.selectbox("State", states, index=states.index("Andhra Pradesh") if "Andhra Pradesh" in states else 0)
seasons = sorted(df[(df["State"] == state) & df["Crop"].isin(CROPS)]["Season"].unique())
if not seasons:
    st.warning("No data for these crops in this state."); st.stop()
season = b.selectbox("Season", seasons)
area = c.number_input("Farm area (ha)", 0.1, 1000.0, 1.0, 0.5)

rows = []
for crop in CROPS:
    s = df[(df["State"] == state) & (df["Season"] == season) & (df["Crop"] == crop) & (df["Yield"] > 0)]
    s = s.sort_values("Crop_Year")
    if s.empty:
        continue
    y, method = float(s["Yield"].iloc[-1]), f"Last recorded yield ({int(s['Crop_Year'].iloc[-1])})"
    if crop == "Rice":
        g = grid[(grid["State"] == state) & (grid["Season"] == season)]
        if len(g):
            gg = g[g["Crop_Year"] == g["Crop_Year"].max()]
            r = gg.loc[(gg["Rain"] - gg["rain0"]).abs().idxmin()]
            y, method = float(r["Forecast"]), "Qiskit hybrid forecast"
    rows.append({"Crop": crop, "Yield (t/ha)": round(y, 2), "Method": method})
if not rows:
    st.warning("No data for this state and season."); st.stop()

st.subheader("Assumptions (edit these)")
st.warning("Water need, cost and price are ILLUSTRATIVE placeholders, not official figures. "
           "Replace them with local data before relying on the profit numbers.")
assume = st.data_editor(DEFAULTS, hide_index=True, disabled=["Crop"], key="assume")

t = pd.DataFrame(rows).merge(assume, on="Crop")
t["Revenue (₹/ha)"] = t["Yield (t/ha)"] * 10 * t["Price (₹/quintal)"]
t["Profit (₹/ha)"] = t["Revenue (₹/ha)"] - t["Cost (₹/ha)"]
t["Total profit (₹)"] = t["Profit (₹/ha)"] * area
t["Profit per 100 mm water (₹/ha)"] = t["Profit (₹/ha)"] / (t["Water need (mm)"] / 100)
t["Source"] = np.where(t["Crop"] == "Rice", "Qiskit hybrid forecast", "Last recorded yield")

m1, m2, m3 = st.columns(3)
m1.metric("Highest profit (under your assumptions)", t.loc[t["Profit (₹/ha)"].idxmax(), "Crop"])
m2.metric("Highest yield", t.loc[t["Yield (t/ha)"].idxmax(), "Crop"])
m3.metric("Lowest water need", t.loc[t["Water need (mm)"].idxmin(), "Crop"])

scale = alt.Scale(domain=["Qiskit hybrid forecast", "Last recorded yield"], range=[GOLD, GREEN])
d1, d2 = st.columns(2)
with d1:
    with st.container(border=True):
        st.subheader("Expected yield (t/ha)")
        st.altair_chart(alt.Chart(t).mark_bar().encode(
            y=alt.Y("Crop:N", sort="-x", title=None), x=alt.X("Yield (t/ha):Q"),
            color=alt.Color("Source:N", scale=scale, legend=alt.Legend(title=None, orient="bottom"))),
            width="stretch")
with d2:
    with st.container(border=True):
        st.subheader("Profit per ha (₹, illustrative)")
        st.altair_chart(alt.Chart(t).mark_bar().encode(
            y=alt.Y("Crop:N", sort="-x", title=None), x=alt.X("Profit (₹/ha):Q"),
            color=alt.Color("Source:N", scale=scale, legend=None)), width="stretch")

show = t[["Crop", "Yield (t/ha)", "Source", "Water need (mm)", "Cost (₹/ha)", "Price (₹/quintal)",
          "Profit (₹/ha)", "Total profit (₹)", "Profit per 100 mm water (₹/ha)"]].copy()
for col in ["Cost (₹/ha)", "Profit (₹/ha)", "Total profit (₹)", "Profit per 100 mm water (₹/ha)"]:
    show[col] = show[col].round(0).astype(int)
st.dataframe(show.sort_values("Profit (₹/ha)", ascending=False), hide_index=True)

st.info("How to read this: rice uses our Qiskit hybrid forecast; other crops use their last recorded "
        "yield, which was as accurate as our models in our tests. Yields are state-level, in t/ha as "
        "reported in the dataset. Crops with unclear units (coconut, onion, cotton) are left out. "
        "Rankings depend entirely on the editable assumptions and are not a recommendation.")