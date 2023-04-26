from Environment.env_reward_win_basic import Environment
from Model.basic_small import Model

import torch.optim as optim
import torch as torch

from time import sleep

load_model = False

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y+11, 8, 0.9, 0.9, -0.00001)
    if load_model: m.net.load_state_dict(torch.load("model_RBS.pt"))
    env = Environment(env_size_x, env_size_y)

    optimizer = optim.Adam(m.parameters(), lr=0.01)

    for epoch in range(100000):
        done = False
        while not done:
            
            done = m.policy(env)
            #sleep(1)
        
        loss = m.train(optimizer)
        print(epoch, loss)
        env.reset()
    
        torch.save(m.net.state_dict(), "model_RBS.pt")
    

if __name__ == '__main__':
    main()