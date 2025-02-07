import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size + hidden_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor, hidden):
        h_prev, c_prev = hidden

        combined = torch.cat((x, h_prev), dim=1)

        gates: torch.Tensor = self.fc(combined)

        i, g, o = gates.chunk(3, dim=1)

        f = torch.sigmoid(c_prev)
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c) * f

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
        self.feature_weighting = nn.Parameter(torch.ones(1, 1, input_size))
        self.temporal_weighting = nn.Parameter(torch.ones(1, seq_length, 1))
        self.magnitude = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, hidden=None):
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        x = x * self.feature_weighting * self.temporal_weighting * self.magnitude.exp()

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
