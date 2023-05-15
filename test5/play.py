from env import Environment
from model import Model

import pygame
import torch as torch

load_model = True

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y*2+9, 8, 0.9, 0.9, -0.000001)
    m.net.load_state_dict(torch.load("model.pt"))
    env = Environment(env_size_x, env_size_y, True)

    while True:
        while env.player == 1:
            m.policy(env)
            #sleep(1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                print(event.key-48)
                env.action(int(event.key-48))
                #sleep(1)
                

if __name__ == '__main__':
    main()