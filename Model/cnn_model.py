from Model.base import ModelBase

from PIL import Image

import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn

import os.path

class CNNModel(ModelBase):
    def __init__(self, in_dim, out_dim, gamma, epsilon, epsilon_rate):
        super().__init__(gamma, epsilon, epsilon_rate)

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Conv2d(3, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(612, 80),
            nn.ReLU(),
            nn.Linear(80, out_dim)
        ]

        self.net = nn.Sequential(*layers)



def test():
    img = Image.open("screenshots/shot.jpg")
    '''img = img.resize((80, 156))
    img.save("screenshots/shot.jpg")
    return'''

    convert_tensor = transforms.ToTensor()

    img = convert_tensor(img)

    layers = [
        nn.Conv2d(3, 1, 3),
        nn.ReLU(),
        nn.MaxPool2d(2),


        #nn.Flatten()
    ]
    
    
    net = nn.Sequential(*layers)
    tensor = net(img)
    print(tensor.shape)
    #return
    
    tensor = tensor*255
    tensor = tensor.detach()
    #print(tensor)
    tensor = np.array(tensor, dtype=np.uint8)
    #print(tensor)
    #tensor_img = [v[0] for v in tensor]
    
    #print(tensor_img)
    tensor = tensor[0]
    #print(tensor)
    Image.fromarray(tensor).save("screenshots/shot_conv.jpg")

