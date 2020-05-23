import torch
import torchvision
import numpy as np

class FCNetwork(torch.nn.Module):
    def __init__(self, hidden_layer_num):
        super(FCNetwork, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(28*28, hidden_layer_num),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(hidden_layer_num, hidden_layer_num),
            torch.nn.ReLU(hidden_layer_num),
            torch.nn.Dropout(p=0.5),

            torch.nn.Linear(hidden_layer_num, 50)
        )

    def forward(self, x):
        _x = x.view(x.shape[0], -1)
        return self.net(_x)

    
class ConvolutionNetwork(torch.nn.Module):
    def __init__(self):
        super(ConvolutionNetwork, self).__init__()
        self.conv_module = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5, 1), # 6 @ 24*24
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 6 @ 12*12
            torch.nn.Dropout(p=0.5),
            torch.nn.Conv2d(6, 16, 5, 1), # 16 @ 8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), # 16 @ 4*4
            torch.nn.Dropout(p=0.5),
        )

        self.fc_module = torch.nn.Sequential(
            torch.nn.Linear(16*4*4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(128, 7)
        )


    def forward(self, input):
        x = self.conv_module(input)
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        x = x.view(-1, dim)
        return self.fc_module(x)
    