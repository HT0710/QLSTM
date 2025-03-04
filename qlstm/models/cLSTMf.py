import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x: torch.Tensor, hidden):
        h_prev, c_prev = hidden

        combined = torch.cat((x, h_prev), dim=1)

        gates: torch.Tensor = self.fc(combined)

        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, dim=1)

        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate + c_prev)
        cell_gate = torch.tanh(cell_gate)
        output_gate = torch.sigmoid(output_gate)

        c_next = forget_gate * c_prev + input_gate * cell_gate
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
        self.efw = nn.Parameter(torch.ones(1, 1, input_size))
        self.hidden = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        if not self.hidden:
            self.hidden = [
                (
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                    torch.zeros(batch_size, self.hidden_size, device=x.device),
                )
                for _ in range(self.num_layers)
            ]

        x = x * self.efw.exp()

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                h, c = self.cells[i](x_t, self.hidden[i])
                self.hidden[i] = (h, c)
                x_t = h
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        self.hidden = [(h.detach(), c.detach()) for h, c in self.hidden]

        return outputs, self.hidden
