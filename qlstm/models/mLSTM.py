from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class mLSTMCell(nn.Module):
    """Implements the mLSTMCell model as described in the xLSTM paper.

    This version replaces custom nn.Parameter weights/biases with standard nn.Linear modules.
    All layers include bias.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.

    Methods:
        forward(x, internal_state): Performs a forward pass of the mLSTMCell model.
        (internal_state is a tuple containing the covariance matrix, normalization state, and stabilization state.)

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initializes the mLSTMCell model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Define the six Linear layers for the six gates.
        # Each layer maps the input to a hidden_size-dimensional output.
        self.W_i = nn.Linear(input_size, hidden_size)  # input gate
        self.W_f = nn.Linear(input_size, hidden_size)  # forget gate
        self.W_o = nn.Linear(input_size, hidden_size)  # output gate
        self.W_q = nn.Linear(input_size, hidden_size)  # query gate
        self.W_k = nn.Linear(input_size, hidden_size)  # key gate
        self.W_v = nn.Linear(input_size, hidden_size)  # value gate

    def forward(
        self,
        x: torch.Tensor,
        internal_state: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the mLSTMCell model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            internal_state (tuple): Tuple containing the covariance matrix (C), normalization state (n),
                                    and stabilization state (m).

        Returns:
            Tuple containing:
                - h_t (torch.Tensor): The output tensor.
                - C_t (torch.Tensor): The updated covariance matrix.
                - n_t (torch.Tensor): The updated normalization state.
                - m_t (torch.Tensor): The updated stabilization state.
        """
        # Unpack the internal state
        C, n, m = internal_state

        # Calculate the pre-activations for the gates
        i_tilda = self.W_i(x)
        f_tilda = self.W_f(x)
        o_tilda = self.W_o(x)
        q_t = self.W_q(x)
        # Scale the key gate by sqrt(hidden_size) for normalization
        k_t = self.W_k(x) / torch.sqrt(
            torch.tensor(self.hidden_size, dtype=x.dtype, device=x.device)
        )
        v_t = self.W_v(x)

        # Gate activations
        i_t = torch.exp(i_tilda)  # Exponential activation for input gate
        f_t = torch.sigmoid(f_tilda)  # Sigmoid for forget gate
        o_t = torch.sigmoid(o_tilda)  # Sigmoid for output gate

        # Stabilization state update
        m_t = torch.max(torch.log(f_t) + m, torch.log(i_t))
        i_prime = torch.exp(i_tilda - m_t)

        # Update covariance matrix and normalization state
        C_t = f_t.unsqueeze(-1) * C + i_prime.unsqueeze(-1) * torch.einsum(
            "bi, bk -> bik", v_t, k_t
        )
        n_t = f_t * n + i_prime * k_t

        # Compute normalization divisor using the diagonal of the product
        normalize_inner = torch.diagonal(torch.matmul(n_t, q_t.T))
        divisor = torch.max(
            torch.abs(normalize_inner), torch.ones_like(normalize_inner)
        )
        h_tilda = torch.einsum("bkj,bj -> bk", C_t, q_t) / divisor.view(-1, 1)

        # Final output
        h_t = o_t * h_tilda

        return h_t, C_t, n_t, m_t

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Initialized covariance matrix and normalization state.
        """
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
            torch.zeros(batch_size, self.hidden_size, **kwargs),
        )


class mLSTM(nn.Module):
    """Implements the mLSTM model as described in the xLSTM paper.

    Attributes:
        input_size (int): The size of the input features.
        hidden_size (int): The size of the hidden state.
        num_layers (int): The number of layers in the model.
        bias (bool): Indicates whether bias is included in the calculations.

    Methods:
        forward(x, hidden_states): Performs a forward pass of the sLSTM model.
        init_hidden(batch_size): Initializes the hidden state of the model.

    References:
        - xLSTM: Extended Long Short-Term Memory
          https://arxiv.org/abs/2405.04517
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool = True,
    ) -> None:
        """
        Initializes the sLSTM.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden state.
            num_layers (int): The number of layers in the model.
            bias (bool, optional): Indicates whether bias is included in the calculations. Default is True.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias

        self.cells = nn.ModuleList(
            [
                mLSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass of the sLSTM.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, input_size) if batch_first is False,
                              or (batch_size, seq_len, input_size) if batch_first is True.
            hidden_states (list, optional): List of hidden states for each layer of the model. If None, hidden states are initialized to zero.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size)
            tuple: Tuple containing the hidden states at each layer and each time step.
        """
        batch_size, seq_length, features_size = x.size()

        # Permute the input tensor if batch_first is True
        if hidden_states is None:
            hidden_states = self.init_hidden(batch_size, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_length):
            x_t = x[:, t, :]
            for i in range(self.num_layers):
                h_t, c_t, n_t, m_t = self.cells[i](x_t, hidden_states[i])
                hidden_states[i] = (c_t, n_t, m_t)
                x_t = h_t
            outputs.append(h_t)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden_states

    def init_hidden(
        self, batch_size: int, **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Initializes the hidden state of the model.

        Args:
            batch_size (int): Batch size of the input tensor.

        Returns:
            list: List containing the initialized hidden states for each layer.
        """
        return [cell.init_hidden(batch_size, **kwargs) for cell in self.cells]
