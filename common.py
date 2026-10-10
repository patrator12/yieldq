import numpy as np, pandas as pd, streamlit as st

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

def crop_yields(df, grid, state, season):
    rows = []
    for crop in CROPS:
        s = df[(df["State"] == state) & (df["Season"] == season) & (df["Crop"] == crop) & (df["Yield"] > 0)]
        s = s.sort_values("Crop_Year")
        if s.empty:
            continue
        y, method = float(s["Yield"].iloc[-1]), "Last recorded yield"
        if crop == "Rice":
            g = grid[(grid["State"] == state) & (grid["Season"] == season)]
            if len(g):
                gg = g[g["Crop_Year"] == g["Crop_Year"].max()]
                r = gg.loc[(gg["Rain"] - gg["rain0"]).abs().idxmin()]
                y, method = float(r["Forecast"]), "Qiskit hybrid forecast"
        rows.append({"Crop": crop, "Yield (t/ha)": round(y, 2), "Source": method})
    return pd.DataFrame(rows)