import pennylane as qml
import torch
import torch.nn as nn


class QLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_qubits=4,
        n_qlayers=1,
        backend="default.qubit",
    ):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.n_vrotations = 3  # Number of ratations for Variational layer.
        self.n_esteps = 1  # Number of steps for Entangling layer pairs.
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits * 4)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        @qml.qnode(self.dev_forget, interface="torch")
        def _circuit_forget(inputs, weights):
            self._VQC(inputs, weights, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_forget]

        @qml.qnode(self.dev_input, interface="torch")
        def _circuit_input(inputs, weights):
            self._VQC(inputs, weights, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_input]

        @qml.qnode(self.dev_update, interface="torch")
        def _circuit_update(inputs, weights):
            self._VQC(inputs, weights, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_update]

        @qml.qnode(self.dev_output, interface="torch")
        def _circuit_output(inputs, weights):
            self._VQC(inputs, weights, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=i)) for i in self.wires_output]

        weight_shapes = {"weights": (self.n_qlayers, self.n_vrotations, self.n_qubits)}

        self.clayer_in = torch.nn.Linear(self.n_inputs + self.hidden_size, n_qubits * 4)
        self.VQC = {
            "forget": qml.qnn.TorchLayer(_circuit_forget, weight_shapes),
            "input": qml.qnn.TorchLayer(_circuit_input, weight_shapes),
            "update": qml.qnn.TorchLayer(_circuit_update, weight_shapes),
            "output": qml.qnn.TorchLayer(_circuit_output, weight_shapes),
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits * 4, self.hidden_size)

    def _ansatz(self, params, wires_type):
        # Entangling layer.
        for k in range(self.n_esteps):
            for i in range(self.n_qubits):
                qml.CNOT(wires=[wires_type[i], wires_type[(i + k + 1) % self.n_qubits]])

        # Variational layer.
        for i in range(self.n_qubits):
            qml.RX(params[0][i], wires=wires_type[i])
            qml.RY(params[1][i], wires=wires_type[i])
            qml.RZ(params[2][i], wires=wires_type[i])

    def _VQC(self, features, weights, wires_type):
        # Pennylane uses batch in second dim
        features = features.transpose(1, 0)

        ry_params = [torch.arctan(feature) for feature in features]
        rz_params = [torch.arctan(feature**2) for feature in features]
        for i in range(self.n_qubits):
            qml.Hadamard(wires=wires_type[i])
            qml.RY(ry_params[i], wires=wires_type[i])
            qml.RZ(rz_params[i], wires=wires_type[i])

        # Variational block.
        qml.layer(self._ansatz, self.n_qlayers, weights, wires_type=wires_type)

    def forward(self, x, hidden=None):
        batch_size, seq_length, features_size = x.size()

        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h_t, c_t = hidden
            h_t = h_t[0]
            c_t = c_t[0]

        outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :]

            v_t = torch.cat((h_t, x_t), dim=1)

            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC["forget"](y_t)))
            i_t = torch.sigmoid(self.clayer_out(self.VQC["input"](y_t)))
            g_t = torch.tanh(self.clayer_out(self.VQC["update"](y_t)))
            o_t = torch.sigmoid(self.clayer_out(self.VQC["output"](y_t)))

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)

        return outputs, (h_t, c_t)
