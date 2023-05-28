import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(Model, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = [
            nn.Linear(in_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_dim)
        ]

        self.net = nn.Sequential(*layers)

        self.clear_batch()

        self.loss_fn = nn.MSELoss()

    def clear_batch(self):
        self.batch = {
            'states' : [], 
            'states_target': []
            }

    def train(self, optimizer):

        loss = self.calc_q_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.clear_batch()
        return loss.item()


    def policy(self, env, debug=False):
        state, state_target = env.get_state()
        state_sampled = torch.from_numpy(np.array(state).astype(np.float32))
        
        actions = self.net(state_sampled)
        
        if debug: 
            print(actions)
            return False

        correct_actions = []
        for i in range(len(state_target)):
            if state_target[i] != -100:
                correct_actions.append(i)
        action = correct_actions[int(np.random.uniform()*len(correct_actions))]
        done = env.action(action)
        self.add_experience(state, state_target)
        return done

    def add_experience(self, state, state_target):
        self.batch['states'].append(state)
        self.batch['states_target'].append(state_target)


    def sample(self):
        for k in self.batch:
            self.batch[k] = np.array(self.batch[k])
            self.batch[k] = torch.from_numpy(self.batch[k].astype(np.float32))


    def calc_q_loss(self):
        self.sample()

        states = self.batch['states']

        states_preds = self.net(states)

        target = self.batch['states_target']

        q_loss = self.loss_fn(states_preds, target)
        return q_loss

