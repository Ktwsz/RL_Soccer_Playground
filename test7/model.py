from model_base import ModelBase

import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn

class Model(ModelBase):
    def __init__(self, in_dim, out_dim, gamma, epsilon, epsilon_rate):
        super().__init__(gamma, epsilon, epsilon_rate)

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Conv2d(1, 1, 4, stride=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(1, 1, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(1, 1, 2, padding=0),
            nn.Flatten(),
            nn.Linear(40, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        ]

        self.net = nn.Sequential(*layers)

