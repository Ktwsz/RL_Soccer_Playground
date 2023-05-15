from model_base import ModelBase

import torch.nn as nn

class Model(ModelBase):
    def __init__(self, in_dim, out_dim, gamma, epsilon, epsilon_rate, lamb):
        super().__init__(gamma, epsilon, epsilon_rate, lamb)

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Linear(in_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_dim)
        ]

        self.net = nn.Sequential(*layers)

