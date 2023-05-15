from env import Environment
from pt_model import Model

import torch as torch

import pygame

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y*9+1, 8)
    m.net.load_state_dict(torch.load("model_pretrained.pt"))

    env = Environment(env_size_x, env_size_y, True)


    while True:
            
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                print(env.get_state())
                env.action(int(event.key-48))
                m.policy(env, True)

    

if __name__ == '__main__':
    main()