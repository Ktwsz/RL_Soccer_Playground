from Environment.env_reward_win_basic import Environment
from Model.basic_large import Model

import torch.optim as optim
import torch as torch

from time import sleep

load_model = False

def main():
    env_size_x = 7
    env_size_y = 13
    m = Model(env_size_x*env_size_y+11, 8, 0.9, 0.9, -0.00001, 0.4)
    if load_model: m.net.load_state_dict(torch.load("model_RBL.pt"))
    env = Environment(env_size_x, env_size_y, False)

    optimizer = optim.Adam(m.parameters(), lr=0.01)

    loss_avg = 0

    for epoch in range(100000):
        done = False
        while not done:
            
            done = m.policy(env)
            #sleep(1)
        
        loss = m.train(optimizer)
        loss_avg += loss
        env.reset()
    
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, average loss is {loss_avg/1000}")
            loss_avg = 0

    torch.save(m.net.state_dict(), "model_RBL.pt")
    

if __name__ == '__main__':
    main()