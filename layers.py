import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True ,w_init_gain='linear'): # noqa 501
        super(Linear, self).__init__()

        self.linear_layer = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, dropout): # noqa 501
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=False)

    def forward(self, x):
        return self.lstm(x)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias, batch_first, dropout): # noqa 501
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            dropout=dropout,
            bidirectional=True)

    def forward(self, x):
        return self.lstm(x)
