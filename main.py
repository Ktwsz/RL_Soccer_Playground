from Environment.env_reward_win_conv import Environment
from time import sleep
import pygame
env = Environment(7, 13, True)

#c = int(input())
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit
        if event.type == pygame.KEYDOWN:
            print(event.key-48)
            env.action(int(event.key-48))

