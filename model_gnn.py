import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNRoutePlanner(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1, activation: str = "leaky_relu"):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)

        # activation function 선택 지원
        self.activation_type = activation.lower()
        self.activation = self._get_activation_fn()

    def _get_activation_fn(self):
        if self.activation_type == "relu":
            return F.relu
        elif self.activation_type == "leaky_relu":
            return lambda x: F.leaky_relu(x, negative_slope=0.01)
        elif self.activation_type == "elu":
            return F.elu
        elif self.activation_type == "silu":
            return F.silu
        else:
            raise ValueError(f"Unsupported activation: {self.activation_type}")

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.conv3(x, edge_index)
        return x
