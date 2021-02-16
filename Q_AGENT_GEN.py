#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:38:14 2021
@author: fabien
"""
import torch, torch.nn as nn
import torch.optim as optim
import numpy as np

from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

TEST_CLASS = True

################################ GRAPH of Network
class Q_AGENT():
    def __init__(self, NB_P_GEN, IO, BS):
        self.P_MIN = 1
        self.P_DILEMNA = 0.4*np.random.random() + 0.1
        # I/O
        self.IO = IO # image cells, action
        self.batch_size = BS
        ## Init
        self.NET = GRAPH_EAT([NB_P_GEN, self.IO[0], self.IO[1], self.P_MIN], None)
        self.NEURON_LIST = self.NET.NEURON_LIST
        self.MODEL = pRNN(self.NEURON_LIST, self.batch_size)
        # nn optimiser
        self.GAMMA = 0.9
        self.optimizer = optim.Adam(self.MODEL.parameters())
        self.criterion = nn.MSELoss()
        ## IO Coordinate
        self.X,self.Y = self.FIRST_IO_COOR_GEN()
        ## Data sample (memory : 'old_state', 'action', 'new_state', 'reward', 'terminal')
        self.MEMORY = [[],[],[],[],[]]
    
    ## Define coordinate of IN/OUT
    def FIRST_IO_COOR_GEN(self) :
        ## Input part
        p = int(np.sqrt(self.IO[0]))+2
        Xa,Xb = np.mod(np.arange(p**2),p)-2, np.repeat(np.arange(p)-2,p)
        X = np.concatenate((Xa[None],Xb[None]), axis = 0).T
        ## Output part
        Y = np.mgrid[-1:2,-1:2].reshape(-1,2)
        ## Get coordinate (no repeat -> replace=False)
        x = np.random.choice(range(len(X)), self.IO[0], replace=False)
        y = np.random.choice(range(len(Y)), self.IO[1], replace=False)
        # note for output min : cyclic 3,4 if 3 mvt, 2 if 4 mvt
        return X[x], Y[y]
    
    ## Action Exploration/Exploitation Dilemna
    def ACTION(self, Input) :
        img_in = torch.tensor(Input, dtype=torch.float)
        DEAL = True #np.random.choice([True,False], 1, p=[self.P_DILEMNA, 1 - self.P_DILEMNA])[0]
        if DEAL :
            # prnn buffer
            output = self.MODEL(img_in)
            # exploration
            next_action = np.random.randint(self.IO[1])
        else :
            # exploitation
            output = self.MODEL(torch.from_numpy(img_in).float())
            # Max values classification
            _, predicted = torch.max(output.data, 1)
            predicted = predicted.numpy()
            next_action = predicted[0]
        return next_action
    
    ## Memory sequencing
    def SEQUENCING(self, prev_state,action,new_state,reward,DONE):
        self.MEMORY[0] += [prev_state]
        self.MEMORY[1] += [action]
        self.MEMORY[2] += [new_state]
        self.MEMORY[3] += [reward]
        self.MEMORY[4] += [DONE]
        if DONE :
            self.MEMORY[0] = torch.tensor(np.concatenate(self.MEMORY[0]), dtype=torch.float)
            self.MEMORY[1] = torch.tensor(np.array(self.MEMORY[1]),  dtype=torch.long).unsqueeze(1)
            self.MEMORY[2] = torch.tensor(np.concatenate(self.MEMORY[2]), dtype=torch.float)
            self.MEMORY[3] = torch.tensor(np.array(self.MEMORY[3]))
            self.MEMORY[4] = torch.tensor(np.array(self.MEMORY[4]), dtype=torch.int)
    
    ## Training Q-Table
    def OPTIM(self) :
        # extract info
        old_state, action, new_state, reward, DONE = self.MEMORY
        # Compute predicted Q-values for each action
        pred_q_values_batch = torch.sum(self.MODEL(old_state).gather(1, action),dim=1).detach()
        pred_q_values_next = self.MODEL(new_state)
        # Compute targeted Q-value for action performed
        target_q_values_batch = reward+(1-DONE)*self.GAMMA*torch.max(pred_q_values_next, 1)[0]
        # zero the parameter gradients
        self.MODEL.zero_grad()
        # Compute the loss
        loss = self.criterion(pred_q_values_batch,target_q_values_batch)
        # Do backward pass
        loss.backward()
        self.optimizer.step()
