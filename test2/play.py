from env_reward_move_large import Environment
from big_input_large import Model

import torch as torch
import pygame

from time import sleep

load_model = True

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y*9+1, 8, 0.9, 0, -0.00001, 0.4)
    if load_model: m.net.load_state_dict(torch.load("model_MLL.pt"))
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