import torch
import torch.nn as nn
import numpy as np
import random
import os

# Setting seeds to ensure reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomGRUCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self.Wz = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uz = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bz = nn.Parameter(torch.Tensor(hidden_dim))

        self.Wr = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Ur = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.br = nn.Parameter(torch.Tensor(hidden_dim))

        self.Wh = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.Uh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.bh = nn.Parameter(torch.Tensor(hidden_dim))

        # Initialize weights using glorot uniform (xavier_uniform_ in PyTorch)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Wz)
        nn.init.xavier_uniform_(self.Uz)
        nn.init.xavier_uniform_(self.Wr)
        nn.init.xavier_uniform_(self.Ur)
        nn.init.xavier_uniform_(self.Wh)
        nn.init.xavier_uniform_(self.Uh)
        nn.init.zeros_(self.bz)
        nn.init.zeros_(self.br)
        nn.init.zeros_(self.bh)

    def forward(self, x, states):
        h_prev = states[0]
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_prev, self.Ur) + self.br)
        z = torch.sigmoid(torch.matmul(x, self.Wz) + torch.matmul(h_prev, self.Uz) + self.bz)
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + r * torch.matmul(h_prev, self.Uh) + self.bh)
        h = (1 - z) * h_prev + z * h_hat
        return h, [h]


class CustomGRULayer(nn.Module):
    def __init__(self, input_dim, hidden_dim,batch_first=False, go_backwards=False, return_sequences=True, return_state=False,
                 stateful=False):
        """

        :param input_dim:
        :param hidden_dim:
        :param batch_first:
        :param go_backwards:
        :param return_sequences:
        :param return_state:
        :param stateful:
        output : hidden : [num_layers * num_directions, batch_size, hidden_size]
        """
        super(CustomGRULayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.go_backwards = go_backwards
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        self.batch_first = batch_first

        self.gru_cell = CustomGRUCell(input_dim, hidden_dim)

    def forward(self, inputs, hidden=None):
        if hidden is None:
            # Automatically create an initial hidden state if none is provided
            batch_size = inputs.size(0) if self.batch_first else inputs.size(1)
            hidden = torch.zeros(batch_size, self.hidden_dim).to(inputs.device)

            # Proceed with your existing forward logic using 'inputs' and 'hidden'
            ...

        # Adjust for batch_first configuration
        if self.batch_first:
            # If batch_first=True, input shape is expected to be (batch, seq, feature)
            inputs = inputs.transpose(0, 1)  # Convert to (seq, batch, feature)

        if self.go_backwards:
            inputs = inputs.flip(dims=[0])  # Flip along the sequence dimension

        if hidden is None:
            batch_size = inputs.size(1)
            hidden_state = torch.zeros(batch_size, self.hidden_dim).to(inputs.device)
        else:
            hidden_state = hidden

        outputs = []
        for t in range(inputs.size(0)):  # Iterate through time steps
            x_t = inputs[t]  # Get the current time step data
            hidden_state, _ = self.gru_cell(x_t.unsqueeze(0), hidden_state.unsqueeze(0))
            outputs.append(hidden_state)

            # Inside your CustomGRU's forward method, before torch.cat(outputs, ...)
        # if not outputs:
        #     # Handle the empty case; return an appropriately sized tensor, perhaps zeros
        #     # Ensure to match the expected shape and device
        #     return torch.zeros(0, batch_size, self.hidden_dim)
        # else:
        #     # Proceed with concatenation when outputs are not empty
        outputs = torch.cat(outputs, dim=0)

        # outputs = torch.cat(outputs, dim=0)
        if not self.return_sequences:
            outputs = outputs[-1]  # Take the last output for each sequence

        if self.batch_first:
            outputs = outputs.transpose(0, 1)  # Convert back to (batch, seq, feature) if necessary



        return outputs, hidden_state

    # Note: In PyTorch, you often don't need to implement the `get_config` method as model saving and loading handle the layers' states differently compared to TensorFlow.

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'go_backwards': self.go_backwards,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'stateful': self.stateful,
            'batch_first': self.batch_first,
        }
        return config
