import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.W_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.b_hh = nn.Parameter(torch.Tensor(3 * hidden_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_ih)
        nn.init.xavier_uniform_(self.W_hh)
        nn.init.zeros_(self.b_ih)
        nn.init.zeros_(self.b_hh)

    def forward(self, x: torch.Tensor, hidden):
        h_prev, c_prev = hidden

        gates = (
            torch.matmul(x, self.W_ih.t())
            + torch.matmul(h_prev, self.W_hh.t())
            + self.b_ih
            + self.b_hh
        )

        input_gate, cell_gate, output_gate = gates.chunk(3, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(c_prev)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = forget_gate * c_prev + input_gate * cell_gate * 2
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, self.hidden_size, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h, c = hidden

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i], c[i] = cell(x_t, (h[i], c[i]))
                x_t = h[i]
            outputs.append(h[-1])

        outputs = torch.stack(outputs, dim=1)
        return outputs, (h, c)
