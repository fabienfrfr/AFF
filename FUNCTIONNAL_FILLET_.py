#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:43:17 2022
@author: fabien
"""
# ML modules
import numpy as np, pandas as pd #, random
import torch, torch.nn as nn
import torch.nn.functional as F

# system module
import pickle, datetime, os

# networks construction
from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

##### Prerequisite
from collections import namedtuple, deque
import itertools

class ReplayMemory(object):
    def __init__(self, capacity, named_tuple):
        self.Transition = named_tuple
        self.memory = deque([],maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))
    def sample(self, batch_size):
        last = self.__len__()
        sample = list(itertools.islice(self.memory, last-batch_size, last))
        return sample
        #return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class CTRL_NET(nn.Module):
    def __init__(self, IO):
        super(CTRL_NET, self).__init__()
        I,O = IO
        if I+O > 64 :
            H = 2*int(np.sqrt(I+O))
        else : 
            H = 16
        self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True)
        self.H1 = nn.Linear(I, H)
        self.H2 = nn.Linear(H, H)
        self.OUT = nn.Linear(H, O)

    def forward(self, x):
        s = x.shape
        x = self.IN(x.view(s[0],s[1],1)).view(s)
        x = F.relu(x)
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
class FunctionnalFillet():
    def __init__(self, arg, NAMED_MEMORY=None, TYPE="class", TIME_DEPENDANT = False):
        # parameter
        self.IO =  arg[0]
        self.BATCH = arg[1]
        self.NB_GEN = arg[2]
        self.NB_SEEDER = arg[3]
        self.NB_EPISOD = arg[4]
        self.NB_E_P_G = int(self.NB_EPISOD/self.NB_GEN)
        self.TIME_DEP = TIME_DEPENDANT
        self.TYPE = TYPE
        self.NAMED_M = NAMED_MEMORY
        # generate first ENN model
        self.GRAPH_LIST = [GRAPH_EAT([self.IO, 1], None) for n in range(self.NB_SEEDER-1)]
        self.SEEDER_LIST = [CTRL_NET(self.IO)]
        for g in self.GRAPH_LIST :
            NEURON_LIST = g.NEURON_LIST
            self.SEEDER_LIST += [pRNN(NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
        # training parameter
        self.NEURON_LIST = []
        self.UPDATE_MODEL()
        # selection
        self.loss = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'N_BATCH', 'LOSS_VALUES'])
        self.supp_param = None
        # evolution param
        self.NB_CONTROL = int(np.rint(np.power(self.NB_SEEDER, 1./4)))
        self.NB_EVOLUTION = int(np.sqrt(self.NB_SEEDER)-1) # square completion
        self.NB_CHALLENGE = int(self.NB_SEEDER - (self.NB_EVOLUTION**2 + self.NB_CONTROL))
        # evolution variable
        self.PARENTING = [-1*np.ones(self.NB_SEEDER)[None]]
        self.PARENTING[0][:self.NB_CONTROL] = 0
        
    def UPDATE_MODEL(self):
        # neuron graph history
        self.NEURON_LIST += [g.NEURON_LIST for g in self.GRAPH_LIST]
        # torch
        self.optimizer = [torch.optim.Adam(s.parameters()) for s in self.SEEDER_LIST]
        if self.TYPE == "class" :
            self.criterion = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
        else :
            self.criterion = [nn.SmoothL1Loss() for n in range(self.NB_SEEDER)] # regression / RL
        # memory
        if self.NAMED_M == None :
            self.memory = {"X_train":None, "Y_train":None, "X_test":None, "Y_test":None}
        else :
            self.memory = [ReplayMemory(1024, self.NAMED_M) for n in range(self.NB_SEEDER)]
        
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
            order = np.argsort(DILEMNA)+1
            #order[np.argmax(order)] += 1
            order = np.exp(order)
            # probability
            p_norm = order/order.sum()
            out_choice = np.random.choice(self.IO[1], p=p_norm)
        return out_choice
    
    def TRAIN(self, output, target, generation=0, index=0, episod=0, i_batch=0):
        # reset
        #self.optimizer[index].zero_grad()
        self.SEEDER_LIST[index].zero_grad()
        # loss computation
        loss = self.criterion[index](output, target)
        # do back-ward
        loss.backward()
        self.optimizer[index].step()
        # save loss
        self.loss = self.loss.append({'GEN':generation,
                                      'IDX_SEED':index,
                                      'EPISOD':episod,
                                      'N_BATCH':i_batch,
                                      'LOSS_VALUES':float(loss.detach().numpy())},
                                      ignore_index=True)
    
    def FIT(self, train_in, train_target, test_in, test_target):
        # SUPERVISED TRAIN
        return
    
    def SELECTION(self, GEN, supp_factor=1):
        # extract data
        sub_loss = self.loss[self.loss.GEN == GEN]
        gb_seed = sub_loss.groupby('IDX_SEED')
        # sup median loss selection
        medianLoss = np.ones(self.NB_SEEDER)
        for i,g in gb_seed :
            median_eps = g.EPISOD.median()
            medianLoss[int(i)] = g[g.EPISOD > median_eps].LOSS_VALUES.mean()
        # normalization
        score = supp_factor*(medianLoss/medianLoss.sum())
        # order
        order = np.argsort(score[self.NB_CONTROL:])
        ### stock control network
        NET_C = self.SEEDER_LIST[:self.NB_CONTROL]
        ### generation parenting
        PARENT = [0]*self.NB_CONTROL
        ### survivor
        GRAPH_S = []
        NET_S = []
        GRAPH_IDX = list(order[:self.NB_EVOLUTION])
        for i in GRAPH_IDX :
            GRAPH_S += [self.GRAPH_LIST[i]]
            if np.random.choice((True,False), 1, [0.9,0.1]):
                NET_S += [self.SEEDER_LIST[self.NB_CONTROL:][i]]
            else :
                NET_S += [pRNN(GRAPH_S[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
                PARENT += [i+1]
        ### mutation
        GRAPH_M = []
        NET_M = []
        for g,j in zip(GRAPH_S,GRAPH_IDX):
            for i in range(self.NB_EVOLUTION):
                GRAPH_M += [g.NEXT_GEN()]
                NET_M += [pRNN(GRAPH_M[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
                PARENT += [j+1]
        ### news random
        GRAPH_N = []
        NET_N = []
        for n in range(self.NB_CHALLENGE):
            GRAPH_N += [GRAPH_EAT([self.IO, 1], None)]
            NET_N += [pRNN(GRAPH_N[-1].NEURON_LIST, self.BATCH, self.IO[0], STACK=self.TIME_DEP)]
            PARENT += [-1]
        ### update seeder list and stock info
        self.PARENTING += [np.array(PARENT)[None]]
        self.GRAPH_LIST = GRAPH_S + GRAPH_M + GRAPH_N
        self.SEEDER_LIST = NET_C + NET_S + NET_M + NET_N
        ### update model
        self.UPDATE_MODEL()
    
    def FINALIZATION(self, supp_param=None, save=True):
        self.PARENTING = np.concatenate(self.PARENTING).T
        self.supp_param = supp_param
        if save :
            if(not os.path.isdir('OUT')): os.makedirs('OUT')
            time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filehandler = open("OUT"+os.path.sep+"MODEL_"+time+".obj", 'wb')
            pickle.dump(self, filehandler); filehandler.close()
    
    def PREDICT(self, INPUT, index=0):
        in_tensor = torch.tensor(INPUT, dtype=torch.float)
        # extract prob
        out_probs = self.SEEDER_LIST[index](in_tensor)
        out_probs = np.squeeze(out_probs.detach().numpy())
        return np.argmax(out_probs)

        
##################################### GYM - TEST
import gym
from tqdm import tqdm
env = gym.make("CartPole-v0")

N_MAX = 300
env._max_episode_steps=N_MAX

NB_OBS = env.observation_space.shape[0]
NB_ACTION = env.action_space.n

IO =  (NB_OBS,NB_ACTION)
BATCH = 25
NB_GEN = 100
NB_SEED = 4**2
NB_EPISODE = 25000 #25000

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
MODEL = FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE], Transition, TYPE='RL')

# loop
duration = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'DURATION'])
select = False
for g in tqdm(range(MODEL.NB_GEN)) :
    # per seeder
    D = []
    for n in range(MODEL.NB_SEEDER):
        # train
        render,l = 0,0
        for i in range(MODEL.NB_E_P_G):
            new_state = env.reset()
            done = False
            # gen data
            j = 0
            while not done :
                action = MODEL.STEP(new_state[None], n)
                state = new_state
                new_state, reward, done, info = env.step(action)
                if done and j < N_MAX-10 :
                    reward = -10
                # see
                if render == 0 :
                    env.render()
                MODEL.memory[n].push(state[None], action, new_state[None], reward, done)
                # iteration
                j+=1
                l+=1
            # duration
            duration = duration.append({'GEN':g, 'IDX_SEED':n,'EPISOD':i,'DURATION':j},ignore_index=True)
            render+=1
            # fit
            if l >= MODEL.BATCH :
                nb_batch = max(1, np.rint(j/MODEL.BATCH).astype(int))
                transitions = MODEL.memory[n].sample(nb_batch*MODEL.BATCH)
                # batch adapted loop
                for b in range(nb_batch) :
                    batch = Transition(*zip(*transitions[b*MODEL.BATCH : (b+1)*MODEL.BATCH]))
                    pred_q, target_q = Q_TABLE(MODEL.SEEDER_LIST[n], batch)
                    MODEL.TRAIN(pred_q, target_q, g, n, i, b)
    # Apply natural selection
    MODEL.SELECTION(g)
    if g == 1 : break
## Finalization
MODEL.FINALIZATION(duration,False)
env.close()




###
print(MODEL.PARENTING)

### plot
import pylab as plt
#from scipy.ndimage import filters
duration = np.array(duration)
#plt.plot(filters.gaussian_filter1d(duration.T,2))
plt.plot(duration.T)

## MNIST
from tensorflow.keras.datasets import mnist
(X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()