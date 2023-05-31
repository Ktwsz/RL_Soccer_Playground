import numpy as np
import torch
import torch.nn as nn
from math import exp

class ModelBase(nn.Module):

    def __init__(self, gamma, epsilon, epsilon_rate):
        super(ModelBase, self).__init__()

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_rate = epsilon_rate

        self.clear_batch()

        self.loss_fn = nn.MSELoss()

    def clear_batch(self):
        self.batch = {
            'states' : [], 
            'rewards': [], 
            'actions': [],
            'next_states': [],
            'dones': []
            }

    def train(self, optimizer):

        loss = self.calc_q_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.clear_batch()
        self.update_epsilon()
        return loss.item()

    def update_epsilon(self):
        self.epsilon -= self.epsilon_rate

    def policy(self, env, debug=False):
        state = env.get_state()
        actions_target = env.get_actions()
        state_sampled = torch.from_numpy(np.array(state).astype(np.float32))
        
        actions = self.net(state_sampled)
        if debug: print(actions)

        correct_actions = []

        for i in range(len(actions_target)):
            if actions_target[i] != -100:
                correct_actions.append(i)
        
        if self.epsilon > np.random.rand():
            action = self.policy_random(correct_actions)
        else:
            action = self.policy_greedy(actions)
        reward, next_state, done = env.action(action)
        self.add_experience(state, action, reward, next_state, done)
        return done

    def policy_greedy(self, actions):
        action_val = -float('inf')
        action_id = 0
        for i in range(actions.size(dim=0)):
            if actions[i].item() > action_val:
                action_val = actions[i].item()
                action_id = i
        return action_id

    def policy_random(self, correct_actions):
        return correct_actions[int(np.random.uniform()*len(correct_actions))]
    
    

    def add_experience(self, state, action, reward, next_state, done):
        self.batch['states'].append(state)
        self.batch['actions'].append(action)
        self.batch['rewards'].append(reward)
        self.batch['next_states'].append(next_state)
        self.batch['dones'].append(done)


    def sample(self):
        for k in self.batch:
            self.batch[k] = np.array(self.batch[k])
            self.batch[k] = torch.from_numpy(self.batch[k].astype(np.float32))

    def calc_q_loss(self):
        act_q_targets = []

        self.sample()

        states = self.batch['states']
        next_states = self.batch['next_states']

        q_preds = self.net(states)
        with torch.no_grad():
            next_q_preds = self.net(next_states)

        act_q_preds = q_preds.gather(-1, self.batch['actions'].long().unsqueeze(-1)).squeeze(-1)
        next_actions = torch.cat((self.batch['actions'][1:], torch.tensor([0])), 0)
        act_next_q_preds = next_q_preds.gather(-1, next_actions.long().unsqueeze(-1)).squeeze(-1)

        act_q_targets = self.batch['rewards'] + self.gamma*(1-self.batch['dones'])*act_next_q_preds

        q_loss = self.loss_fn(act_q_preds, act_q_targets)
        return q_loss

