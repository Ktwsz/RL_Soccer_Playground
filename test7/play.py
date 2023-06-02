from env import Environment
from model import Model

import pygame
import torch as torch

moves = {113: 4, 119: 2, 101: 6, 97: 0, 100: 1, 122: 5, 120: 3, 99: 7}

load_model = True

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y*2+9, 8, 0.9, 0, -0.000001)
    m.net.load_state_dict(torch.load("model.pt"))
    env = Environment(env_size_x, env_size_y, True)

    done = False
    while not done:
        while env.player == 1:
            done = m.policy(env)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key in list(moves.keys()):
                    r, s, done = env.action(moves[event.key])
                else:
                    r, s, done = env.action(int(event.key-48))
                

if __name__ == '__main__':
    main()