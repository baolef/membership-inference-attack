import torch.nn as nn

from models import register


@register('mlp')
class MLP(nn.Module):
    def __init__(self, n_input, layers, activation, dropout):
        super().__init__()
        layers = [n_input] + layers
        network = []
        activation = nn.__dict__[activation]
        for i in range(len(layers) - 1):
            network.append(nn.Linear(layers[i], layers[i + 1]))
            network.append(nn.BatchNorm1d(layers[i + 1]))
            network.append(activation())
            network.append(nn.Dropout(dropout))
        network.append(nn.Linear(layers[-1], 1))  # binary classification
        network.append(nn.Sigmoid())
        self.model = nn.Sequential(*network)

    def forward(self, x):
        y = self.model(x)
        return y
