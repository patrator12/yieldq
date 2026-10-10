import sys, json, time
import numpy as np, pandas as pd
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector

STATE = sys.argv[1] if len(sys.argv) > 1 else "Andhra Pradesh"
SEASON = sys.argv[2] if len(sys.argv) > 2 else "Kharif"
AREA, WATER_AVG, BUDGET_HA, MAXC = 10.0, 700.0, 80000.0, 3
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

CROPS = ["Rice", "Maize", "Groundnut", "Dry chillies", "Jowar", "Gram", "Sunflower"]
WATER = [1200, 600, 550, 800, 450, 350, 500]
COST = [60000, 50000, 55000, 150000, 30000, 30000, 35000]
PRICE = [2300, 2200, 6000, 15000, 3300, 5500, 6500]

df = pd.read_csv("data/crop_yield.csv")
for c in ["Crop", "Season", "State"]:
    df[c] = df[c].str.strip()
grid = pd.read_csv("data/forecast_grid.csv")

names, yld = [], []
for crop in CROPS:
    s = df[(df["State"] == STATE) & (df["Season"] == SEASON) & (df["Crop"] == crop) & (df["Yield"] > 0)]
    s = s.sort_values("Crop_Year")
    if s.empty:
        continue
    y = float(s["Yield"].iloc[-1])
    if crop == "Rice":
        g = grid[(grid["State"] == STATE) & (grid["Season"] == SEASON)]
        if len(g):
            gg = g[g["Crop_Year"] == g["Crop_Year"].max()]
            y = float(gg.loc[(gg["Rain"] - gg["rain0"]).abs().idxmin(), "Forecast"])
    names.append(crop); yld.append(y)
idx = [CROPS.index(c) for c in names]
w, cost, price, yld = np.array(WATER)[idx], np.array(COST)[idx], np.array(PRICE)[idx], np.array(yld)
prof = yld * 10 * price - cost
K, N = len(names), 2 ** len(names)
block = AREA / MAXC
W, B = WATER_AVG * AREA, BUDGET_HA * AREA
print(f"{STATE} / {SEASON}: {K} crops -> {K} qubits, {N} possible choices")

bits = (np.arange(N)[:, None] >> np.arange(K)) & 1          # bit j = qubit j
profit = bits @ (block * prof)
feas = ((bits @ (block * w)) <= W + 1e-9) & ((bits @ (block * cost)) <= B + 1e-9) & (bits.sum(1) <= MAXC)
cost_diag = np.where(feas, -profit, np.abs(block * prof).sum())
cd = cost_diag / np.abs(cost_diag).max()
exact = int(np.argmax(np.where(feas, profit, -np.inf)))
label = lambda i: " + ".join(names[j] for j in range(K) if (i >> j) & 1) or "(none)"
print("exact best:", label(exact), "| profit", round(float(profit[exact])))

# cost function as a sum of Z-strings (Walsh-Hadamard expansion)
z = 1 - 2 * bits
labels, coeffs = [], []
for S in range(N):
    sign = np.ones(N)
    for j in range(K):
        if (S >> j) & 1:
            sign = sign * z[:, j]
    c = float((cd * sign).mean())
    if abs(c) > 1e-12:
        labels.append("".join("Z" if (S >> (K - 1 - pos)) & 1 else "I" for pos in range(K)))
        coeffs.append(c)
HC = SparsePauliOp(labels, np.array(coeffs))

def fast_state(params):
    g, b = params[:REPS], params[REPS:]
    psi = np.full(N, 1 / np.sqrt(N), dtype=complex)
    for l in range(REPS):
        psi = psi * np.exp(-1j * g[l] * cd)
        c, s = np.cos(b[l]), -1j * np.sin(b[l])
        psi = psi.reshape([2] * K)
        for ax in range(K):
            a0, a1 = np.take(psi, 0, axis=ax), np.take(psi, 1, axis=ax)
            psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
        psi = psi.reshape(N)
    return psi

energy = lambda p: float((np.abs(fast_state(p)) ** 2) @ cd)
rng = np.random.default_rng(7)
best = None
for r in range(40):
    res = minimize(energy, rng.uniform(0, np.pi, 2 * REPS), method="COBYLA", options={"maxiter": 800})
    if best is None or res.fun < best.fun:
        best = res
params = best.x
print("optimised energy:", round(best.fun, 4))

# run the real Qiskit circuit with the best parameters and compare
gam = [Parameter(f"g{l}") for l in range(REPS)]
bet = [Parameter(f"b{l}") for l in range(REPS)]
qc = QuantumCircuit(K); qc.h(range(K))
for l in range(REPS):
    qc.append(PauliEvolutionGate(HC, time=gam[l]), range(K))
    for q in range(K):
        qc.rx(2 * bet[l], q)
t0 = time.time()
probs_q = Statevector(qc.assign_parameters(dict(zip(gam + bet, params)))).probabilities()
probs_f = np.abs(fast_state(params)) ** 2
diff = float(np.abs(probs_q - probs_f).max())
print(f"Qiskit circuit vs fast simulation: max difference {diff:.2e} ({time.time()-t0:.1f}s)")
probs = probs_q

order = np.argsort(-probs)[:6]
top = [{"crops": label(int(i)), "prob": float(probs[i]), "feasible": bool(feas[i]),
        "profit": float(profit[i]) if feas[i] else None} for i in order]
for t in top:
    print(f"  {t['prob']*100:5.1f}%  feasible={t['feasible']!s:5}  {t['crops']}")
out = {"state": STATE, "season": SEASON, "area": AREA, "water_avg": WATER_AVG, "budget_ha": BUDGET_HA,
       "max_crops": MAXC, "reps": REPS, "qubits": K, "choices": N,
       "exact": {"crops": label(exact), "profit": float(profit[exact])},
       "p_exact": float(probs[exact]), "p_feasible": float(probs[feas].sum()),
       "random_p_exact": 1 / N, "random_p_feasible": float(feas.mean()),
       "random_profit_ratio": float(np.where(feas, profit, 0).mean() / profit[exact]), "expected_profit_ratio": float((probs * np.where(feas, profit, 0)).sum() / profit[exact]),
       "most_likely_is_exact": bool(order[0] == exact), "verified_max_diff": diff, "top": top}
json.dump(out, open("data/qaoa_result.json", "w"), indent=1)
print("P(exact best):", round(out["p_exact"] * 100, 1), "% vs random", round(100 / N, 1), "%",
      "| P(feasible):", round(out["p_feasible"] * 100, 1), "%",
      "| expected profit ratio:", round(out["expected_profit_ratio"], 2))
print("saved data/qaoa_result.json")