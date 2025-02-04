import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ih = nn.Linear(input_size, 3 * hidden_size)
        self.hh = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor, hidden):
        h_prev, c_prev = hidden

        gates: torch.Tensor = self.ih(x) + self.hh(h_prev)

        i, c, o = gates.chunk(3, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(c_prev)
        c = torch.tanh(c)
        o = torch.sigmoid(o)

        c = f * c_prev + i * c
        h = o * torch.tanh(c)

        return h, c


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.cells = nn.ModuleList(
            [
                LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ]
        )
        self.features_weighting = nn.Parameter(torch.ones(1, 1, input_size))
        self.times_weighting = nn.Parameter(torch.ones(1, seq_length, 1))

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        x = x * self.features_weighting * self.times_weighting

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                h, c = self.cells[i](x_t, hidden[i])
                hidden[i] = (h, c)
                x_t = h
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden
