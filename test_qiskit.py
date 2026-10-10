import qiskit, qiskit_machine_learning, torch
print(qiskit.__version__, qiskit_machine_learning.__version__, torch.__version__)

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

n = 4
fm = ZZFeatureMap(n, reps=1)
an = RealAmplitudes(n, reps=2)
qc = QuantumCircuit(n)
qc.compose(fm, inplace=True)
qc.compose(an, inplace=True)

qnn = EstimatorQNN(circuit=qc, input_params=fm.parameters, weight_params=an.parameters)
model = TorchConnector(qnn)

x = torch.rand(5, n)
y = model(x)
print("output shape:", y.shape)
y.sum().backward()
print("backward pass OK")