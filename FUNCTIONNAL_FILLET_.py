#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:43:17 2022
@author: fabien
"""
# ML modules
import numpy as np #, random
import torch, torch.nn as nn
import torch.nn.functional as F

##### Prerequisite
from collections import namedtuple, deque
import itertools
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        last = self.__len__()
        return list(itertools.islice(self.memory, last-batch_size, last))
        #return random.sample(self.memory, batch_size)
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

## For Q-Learning
def Q_TABLE(model, batch, GAMMA = 0.9):
    old_state = torch.tensor(np.concatenate(batch.state), dtype=torch.float)
    action = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
    new_state = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float)
    reward = torch.tensor(np.array(batch.reward), dtype=torch.long)
    done = torch.tensor(np.array(batch.done), dtype=torch.int)
    # actor proba
    actor = model(old_state)
    # Compute predicted Q-values for each action
    pred_q_values_batch = actor.gather(1, action)
    pred_q_values_next  = model(new_state)
    # Compute targeted Q-value for action performed
    target_q_values_batch = (reward+(1-done)*GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
    # return y, y_prev
    return pred_q_values_batch,target_q_values_batch

##### FF MODULE
class model():
    def __init__(self, arg, TYPE="class", TIME_DEPENDANT = False):
        # parameter
        self.IO =  arg[0]
        self.BATCH_SIZE = arg[1]
        self.NB_GEN = arg[2]
        self.NB_SEEDER = arg[3]
        # generate first ENN model
        self.SEEDER_LIST = [CTRL_NET(self.IO)]
        # training
        self.optimizer = [torch.optim.Adam(self.SEEDER_LIST[0].parameters())]
        if TYPE == "class" :
            self.criterion = [nn.CrossEntropyLoss()]
        else :
            self.criterion = [nn.SmoothL1Loss()] # regression / RL
        # memory
        self.memory = [ReplayMemory(100000)]
    
    def STEP(self, INPUT, index=0) :
        in_tensor = torch.tensor(INPUT, dtype=torch.float)
        out_probs = self.SEEDER_LIST[index](in_tensor)
        # exploration dilemna
        DILEMNA = np.squeeze(out_probs.detach().numpy())
        if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
            out_choice = np.random.randint(self.IO[1])
        else :
            if DILEMNA.min() < 0 : DILEMNA = DILEMNA-DILEMNA.min() # order garanty
            ## ADD dispersion between near values (ex : q-table, values is near)
            order = np.exp(np.argsort(DILEMNA)+1)
            # probability
            p_norm = order/order.sum()
            out_choice = np.random.choice(self.IO[1], p=p_norm)
        return out_choice
    
    def FIT(self, output, target, index=0):
        # reset
        #self.optimizer[index].zero_grad()
        self.SEEDER_LIST[index].zero_grad()
        # loss computation
        loss = self.criterion[index](output, target)
        # do back-ward
        loss.backward()
        self.optimizer[index].zero_grad()
    
    def EVOLUTION(self):
        return

##################################### GYM - TEST
import gym
from tqdm import tqdm
env = gym.make("CartPole-v0")

NB_OBS = env.observation_space.shape[0]
NB_ACTION = env.action_space.n

IO =  (NB_OBS,NB_ACTION)
BATCH = 25
NB_GEN = 100
NB_SEED = 1
NB_EPISODE = 50000

MODEL = model([IO, BATCH, NB_GEN, NB_SEED], TYPE='RL')
Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))

NB_E_P_G = int(NB_EPISODE/NB_GEN)
# loop
duration = []
for g in tqdm(range(MODEL.NB_GEN)) :
    # per seeder
    for n in range(MODEL.NB_SEEDER):
        # train
        r,l = 0,0
        for i in range(NB_E_P_G):
            new_state = env.reset()
            done = False
            # gen data
            j = 0
            while not done :
                action = MODEL.STEP(new_state[None], n)
                state = new_state
                new_state, reward, done, info = env.step(action)
                if done :
                    reward = -10
                # see
                if r == 0 :
                    env.render()
                MODEL.memory[n].push(state[None], action, new_state[None], reward, done)
                # iteration
                j+=1
                l+=1
            # duration
            duration += [j]
            r+=1
            # fit
            if l >= MODEL.BATCH_SIZE :
                nb_batch = max(1, np.rint(j/MODEL.BATCH_SIZE).astype(int))
                transitions = MODEL.memory[n].sample(nb_batch*MODEL.BATCH_SIZE)
                # batch adapted loop
                for b in range(nb_batch) :
                    batch = Transition(*zip(*transitions[b*MODEL.BATCH_SIZE : (b+1)*MODEL.BATCH_SIZE]))
                    pred_q, target_q = Q_TABLE(MODEL.SEEDER_LIST[n], batch)
                    MODEL.FIT(pred_q, target_q, n)
    # Apply natural selection
    MODEL.EVOLUTION()
env.close()          

### plot
import pylab as plt
plt.plot(duration)