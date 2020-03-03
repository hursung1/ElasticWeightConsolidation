import torch
import torchvision
import numpy as np

class FCNetwork(torch.nn.Module):
    def __init__(self, hidden_layer_num):
        super(FCNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, hidden_layer_num),
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_layer_num, hidden_layer_num),
            torch.nn.ReLU(hidden_layer_num),

            torch.nn.Linear(hidden_layer_num, 10)
        )

    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.net(_x)

