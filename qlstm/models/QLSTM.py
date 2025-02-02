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
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        @qml.qnode(self.dev_forget, interface="torch")
        def _circuit_forget(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        @qml.qnode(self.dev_input, interface="torch")
        def _circuit_input(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_input)
            qml.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        @qml.qnode(self.dev_update, interface="torch")
        def _circuit_update(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_update)
            qml.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        @qml.qnode(self.dev_output, interface="torch")
        def _circuit_output(inputs, weights):
            qml.AngleEmbedding(inputs, wires=self.wires_output)
            qml.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.n_inputs + self.hidden_size, n_qubits)
        self.VQC = {
            "forget": qml.qnn.TorchLayer(_circuit_forget, weight_shapes),
            "input": qml.qnn.TorchLayer(_circuit_input, weight_shapes),
            "update": qml.qnn.TorchLayer(_circuit_update, weight_shapes),
            "output": qml.qnn.TorchLayer(_circuit_output, weight_shapes),
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

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
