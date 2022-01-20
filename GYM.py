#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:40:35 2022
@author: fabien
"""
import gym, random
import numpy as np

import torch, torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

env = gym.make("CartPole-v0")

from collections import namedtuple, deque
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class CTRL_NET(nn.Module):
    def __init__(self, IO):
        super(CTRL_NET, self).__init__()
        I,O = IO
        if I+O > 64 :
            H = int(np.sqrt(I+O))
        else : 
            H = int(I+O)
        self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True)
        self.H1 = nn.Linear(I, H)
        self.H2 = nn.Linear(H, H)
        self.OUT = nn.Linear(H, O)

    def forward(self, x):
        s = x.shape
        x = F.relu(self.IN(x.view(s[0],s[1],1)).view(s))
        x = F.relu(self.H1(x))
        x = F.relu(self.H2(x))
        return self.OUT(x)

class Q_AGENT():
    def __init__(self, NB_OBS,NB_ACTION):
        self.IO = NB_OBS,NB_ACTION
        self.MODEL = CTRL_NET((NB_OBS,NB_ACTION))
        self.GAMMA = 0.9
        self.optimizer = torch.optim.Adam(self.MODEL.parameters())
        self.criterion = nn.SmoothL1Loss()

    ## Action Exploration/Exploitation Dilemna
    def ACTION(self, Input) :
        img_in = torch.tensor(Input, dtype=torch.float)
        # actor-critic (old version)
        action_probs = self.MODEL(img_in)
        # exploration-exploitation dilemna
        DILEMNA = np.squeeze(action_probs.detach().numpy())
        if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
            next_action = np.random.randint(self.IO[1])
        else :
            if DILEMNA.min() < 0 : DILEMNA = DILEMNA-DILEMNA.min() # n-1 choice restriction
            ## ADD dispersion between near values (q-table, values is near)
            order = np.square(np.argsort(DILEMNA)+1)
            # probability
            p_norm = order/order.sum()
            print(p_norm)
            next_action = np.random.choice(self.IO[1], p=p_norm)
        return next_action

    ## Q-Table
    def Q_TABLE(self, old_state, action, new_state, reward, DONE) :
        # actor proba
        actor = self.MODEL(old_state)
        # Compute predicted Q-values for each action
        pred_q_values_batch = actor.gather(1, action)
        pred_q_values_next  = self.MODEL(new_state)
        # Compute targeted Q-value for action performed
        target_q_values_batch = (reward+(1-DONE)*self.GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
        # return y, y_prev
        return pred_q_values_batch,target_q_values_batch

NB_ACTION = env.action_space.n
NB_OBS = env.observation_space.shape[0]

AGENT = Q_AGENT(NB_OBS,NB_ACTION)
memory = ReplayMemory(10000)

for i in tqdm(range(20000)) :
    new_state = env.reset()
    done = False
    i = 0
    while not done :
        action = AGENT.ACTION(new_state[None])
        #print(action)
        state = new_state
        new_state, reward, done, info = env.step(action)
        # see
        env.render()
        # Store the transition in memory
        memory.push(state[None], action, new_state[None], reward, done)
        i+=1
    # last memory
    transitions = memory.sample(i) ## !! if not time dependant !!!
    batch = Transition(*zip(*transitions))
    # extrat batch
    old_state = torch.tensor(np.concatenate(batch.state), dtype=torch.float)
    action = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
    new_state = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float)
    reward = torch.tensor(np.array(batch.reward), dtype=torch.long)
    DONE = torch.tensor(np.array(batch.done), dtype=torch.int)
    # Qtable
    pred_q_values_batch,target_q_values_batch = AGENT.Q_TABLE(old_state, action, new_state, reward, DONE)
    # train
    AGENT.MODEL.zero_grad()
    #AGENT.optimizer.zero_grad()
    # Compute the loss
    loss = AGENT.criterion(pred_q_values_batch,target_q_values_batch).type(torch.float)
    # Do backward pass
    loss.backward()
    AGENT.optimizer.step()
env.close()