import pennylane as qml
import torch
import torch.nn as nn


class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        seq_length,
        n_qubits=4,
        n_qlayers=1,
        backend="default.qubit",
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.n_qubits = n_qubits * 3
        self.n_qlayers = n_qlayers
        self.n_vrotations = 3  # Number of ratations for Variational layer
        self.n_esteps = 1  # Number of steps for Entangling layer pairs
        self.backend = backend

        # Weighted matrix
        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))
        self.temporal_weighting = nn.Parameter(torch.ones(1, seq_length, 1))
        self.magnitude = nn.Parameter(torch.tensor(1.0))

        # Accessway
        self.entry = nn.Linear(input_size + hidden_size, self.n_qubits)
        self.exit = nn.Linear(self.n_qubits, self.hidden_size * 3)

        # Variational quantum circuit
        self.wires = [i for i in range(self.n_qubits)]
        self.device = qml.device(self.backend, wires=self.n_qubits)

        @qml.qnode(self.device, interface="torch")
        def _qnode(inputs, weights):
            # Pennylane uses batch in second dim
            features = inputs.transpose(1, 0)

            # Encoding
            ry_params = [torch.arctan(feature) for feature in features]
            rz_params = [torch.arctan(feature**2) for feature in features]
            for i in range(self.n_qubits):
                qml.Hadamard(wires=self.wires[i])
                qml.RY(ry_params[i], wires=self.wires[i])
                qml.RZ(rz_params[i], wires=self.wires[i])

            # Variational block
            qml.layer(self._ansatz, self.n_qlayers, weights, wires_type=self.wires)

            # Measurement
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires]

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}
        self.VQC = qml.qnn.TorchLayer(_qnode, weight_shapes)

    def _ansatz(self, params, wires_type):
        # Entangling layer
        for k in range(self.n_esteps):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % self.n_qubits]])

        # Variational layer
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size, seq_length, features_size = x.size()

        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = hidden
            h_t = h[0]
            c_t = c[0]

        weighted_matrix = (
            self.feature_weighting * self.temporal_weighting * self.magnitude.exp()
        )

        x = x * weighted_matrix

        outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :]

            combined = torch.cat((x_t, h_t), dim=1)

            qubits = self.entry(combined)

            vqc = self.VQC(qubits)

            gates = self.exit(vqc)

            i, g, o = gates.chunk(chunks=3, dim=1)

            f_t = torch.sigmoid(c_t)
            i_t = torch.sigmoid(i)
            g_t = torch.tanh(g)
            o_t = torch.sigmoid(o)

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t) * f_t

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)

        return outputs, (h_t, c_t)
