import torch
from torch import nn


class GRUNet(nn.Module):
    def __init__(self, config_dict):
        super().__init__()
        self.gru = nn.GRU(
            input_size=config_dict["input_dim"],
            hidden_size=config_dict["hidden_dim"],
            num_layers=config_dict["num_hidden"],
            batch_first=True,
            dropout=config_dict["drop_prob"],
            bidirectional=True,
        )
        self.clf = nn.Linear(config_dict["hidden_dim"] * 2, 2)

    def forward(self, x):
        x = x.float()
        return self.clf(self.gru(x)[0][:, -1, :])


class MLP(nn.Module):
    def __init__(self, config_dict):
        super().__init__()

        dims = (
            [config_dict["input_dim"]]
            + [config_dict["hidden_dim"]] * config_dict["num_hidden"]
            + [2]
        )

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if (i + 1) != len(dims) - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=config_dict["drop_prob"]))
                layers.append(nn.BatchNorm1d(config_dict["hidden_dim"]))
        self.clf = nn.Sequential(*layers)

    def forward(self, x):
        x = x.float()
        return self.clf(x)
