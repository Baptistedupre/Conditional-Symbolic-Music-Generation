import torch
import torch.nn as nn 


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, dropout): # noqa 501
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers


    def forward(self, x):
        return self.lstm(x)