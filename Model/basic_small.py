from Model.base import ModelBase

import numpy as np
import torch
import torch.nn as nn

class Model(ModelBase):
    def __init__(self, in_dim, out_dim, gamma, epsilon, epsilon_rate):
        super().__init__(gamma, epsilon, epsilon_rate)

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Linear(in_dim, 80),
            nn.ReLU(),
            nn.Linear(80, out_dim)
        ]

        self.net = nn.Sequential(*layers)


