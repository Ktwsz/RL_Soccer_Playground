from base_lambda import ModelBase

import numpy as np
import torch
import torch.nn as nn

class Model(ModelBase):
    def __init__(self, in_dim, out_dim, gamma, epsilon, epsilon_rate, lamb):
        super().__init__(gamma, epsilon, epsilon_rate, lamb)

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Linear(in_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Linear(200, out_dim),
            nn.Softmax(dim=0)
        ]

        self.net = nn.Sequential(*layers)

