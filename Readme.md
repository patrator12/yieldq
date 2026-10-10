# YieldQ: quantum-hybrid rice yield forecast

Qiskit Fall Fest 2026 · Team: Error Found · Problem statement: VNQFF-03, Quantum-Enhanced Crop Yield Prediction

## Proposed idea
Forecast next-season rice yield (t/ha) for an Indian state. A classical PyTorch encoder compresses 44 prepared features into 4 numbers. A 4-qubit Qiskit circuit turns them into a predicted change in yield. Forecast = last season's yield + predicted change. The app also has a crop comparison page, a water-limited planner, and a small QAOA crop-selection experiment.

## Team
| Name | Role |
|---|---|
| [Name] | [what this person actually did] |
| [Name] | [what this person actually did] |
| [Name] | [what this person actually did] |

Note: [State honestly what was explored before the event, if anything, and what was built during it.]

## Qiskit implementation
- Library versions: Qiskit 2.5.2, qiskit-machine-learning 0.9.1, PyTorch 2.14.1, Python 3.14
- Algorithm: variational quantum circuit trained as a quantum neural network (`EstimatorQNN`) connected to PyTorch with `TorchConnector`
- Circuit: 4 qubits, RY angle encoding, `RealAmplitudes` (2 repetitions, 12 trainable angles), Z measurement on each qubit, run on Qiskit's statevector simulator
- Training: Adam, MSE loss, epochs chosen on validation years 2015-2016
- Crop selection: QAOA with 7 qubits and 5 layers, built from `PauliEvolutionGate` and `Statevector`, angles tuned with COBYLA, checked against an exact search and against a numpy simulation
- First attempt (ZZFeatureMap and one combined readout) did not learn; we changed to RY encoding and per-qubit readout

## Dataset
Kaggle, "Crop Yield in Indian States" (30 states, 1997-2020). We use the rice rows: 1,191 after cleaning, 1,114 with last-year features. Train 1997-2014, validation 2015-2016, test 2017-2019. The data file is not included; download it from Kaggle [add link] and save it as `data/crop_yield.csv`. [Add the dataset's license once you have checked it.]

## Results (test years 2017-2019, mean of 3 seeds, RMSE in t/ha)
| Model | RMSE | R² |
|---|---|---|
| Same as last year | 0.260 | 0.873 |
| Ridge (predicts change) | 0.275 | 0.858 |
| Encoder + classical head | 0.256 ± 0.011 | 0.877 |
| Encoder + Qiskit circuit | 0.266 ± 0.021 | 0.867 |

QAOA test (Andhra Pradesh, Kharif, 5 layers): valid plans 89% (random 40%), expected profit 69% of the best (random 14%), exact best plan 2.5% (random 0.8%).

## Observations and limitations
- The quantum model matches the classical ones; we do not claim a quantum advantage.
- "Same as last year" is a very strong baseline on state-level data.
- Fertilizer and rainfall what-ifs were unreliable and were removed.
- Cost, price and water figures in the app are illustrative placeholders.
- Runs on a simulator only. Next: IBM Quantum hardware and field-level soil, irrigation and satellite data.

## How to run
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    # put the Kaggle file at data/crop_yield.csv
    python prepare_data.py && python baseline.py && python lags.py
    python hybrid4.py 8 42
    python precompute.py
    python -m streamlit run app3.py
    python qaoa.py "Andhra Pradesh" Kharif 5

## Demo and slides
- Slides: `docs/YieldQ_QFF2026_PPT_v3.pptx`
- Screenshots: `docs/screenshots/`

## References and acknowledgements
- Qiskit and qiskit-machine-learning documentation, IBM Quantum
- Havlíček et al. (2019), Nature 567
- Khaki and Wang (2019), Frontiers in Plant Science 10
- PyTorch, scikit-learn, Streamlit
- [If you used an AI assistant for guidance or code drafting, say so here.]