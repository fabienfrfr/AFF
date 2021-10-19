#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:38:14 2021
@author: fabien
"""
import torch, torch.nn as nn
import numpy as np, pylab as plt

from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

################################ AGENT
class Q_AGENT():
    def __init__(self, *arg, CTRL=False, NET = None, DENSITY_IO = None, COOR = None):
        self.P_MIN = 1
        # Parameter
        self.ARG = arg
        self.IO = arg[0] # image cells, action
        self.NB_P_GEN = arg[1]
        self.batch_size = arg[2]
        self.N_TIME = arg[4]
        self.DENSITY_IO = DENSITY_IO
        ## Init
        if CTRL :
            # I/O minimalisme
            X = np.array([[0,0],[0,2],[0,4],[2,0],[2,2],[2,4],[4,0],[4,2],[4,4]])-[2,2]
            Y = np.array([[0,1],[1,2],[2,0]])-[1,1]
            COOR = (X,Y)
            self.NET = GRAPH_EAT(None, self.CONTROL_NETWORK())
        elif NET == None :
            self.NET = GRAPH_EAT([self.NB_P_GEN, self.IO[0], self.IO[1], self.P_MIN], None)
        else :
            self.NET = NET
        self.NEURON_LIST = self.NET.NEURON_LIST
        self.MODEL = pRNN(self.NEURON_LIST, self.batch_size, self.IO[0])
        # nn optimiser
        self.GAMMA = 0.9
        #self.optimizer = torch.optim.Adam(self.MODEL.parameters())
        self.optimizer = torch.optim.SGD(self.MODEL.parameters(), lr=1e-6, momentum=0.9)
        self.criterion = nn.MSELoss(reduction='sum')
        self.loss = None
        ## IO Coordinate
        self.CC = np.mgrid[-2:3,-2:3].reshape(-1,2), np.mgrid[-1:2,-1:2].reshape(-1,2) #complete coordinate
        if COOR == None :
            if DENSITY_IO == None :
                self.X,self.Y = self.FIRST_IO_COOR_GEN()
            else :
                self.X,self.Y = self.FIRST_IO_COOR_GEN(DENSITY_IO)
        else :
            self.X,self.Y = COOR
        ## Data sample (memory : 'old_state', 'action', 'new_state', 'reward', 'terminal')
        self.MEMORY = [[],[],[],[],[]]
        self.MEMORY_ = None
        ## Players
        self.prev_state = None
    
    def INIT_ENV(self, ENV_INPUT) :
        self.prev_state = ENV_INPUT.RESET()
        
    def PARTY(self, ENV_INPUT):
        for t in range(self.N_TIME):
            for i in range(self.batch_size):
                action = self.ACTION(self.prev_state)
                new_state, reward, DONE = ENV_INPUT.STEP(action)
                # Memory update
                if i == self.batch_size-1 : DONE = True
                self.SEQUENCING(self.prev_state,action,new_state,reward,DONE)
                # n+1
                self.prev_state = new_state.copy()
                # escape loop
                if DONE == True : break                
            # Reinforcement learning
            self.OPTIM()
    
    ## Define coordinate of IN/OUT
    def FIRST_IO_COOR_GEN(self, DENSITY = None) :
        ## Density
        if (DENSITY == None) :
            p_X, p_Y = None, None
        else :
            p_X, p_Y = DENSITY
            p_X, p_Y = p_X.reshape(-1), p_Y.reshape(-1)
        ## Get coordinate (no repeat -> replace=False)
        x = np.random.choice(range(len(self.CC[0])), self.IO[0], p = p_X, replace=False)
        y = np.random.choice(range(len(self.CC[1])), self.IO[1], p = p_Y, replace=False)
        # note for output min : cyclic 3,4 if 3 mvt, 2 if 4 mvt
        return self.CC[0][x], self.CC[1][y]
    
    def MUTATION_IO(self, DENSITY):
        p_X, p_Y = DENSITY
        p_X, p_Y = p_X.reshape(-1), p_Y.reshape(-1)
        ## Get coordinate (no repeat -> replace=False)
        x = np.random.choice(range(len(self.CC[0])), 1, p = p_X, replace=False)
        y = np.random.choice(range(len(self.CC[1])), 1, p = p_Y, replace=False)
        # test if included
        TEST_X = ((self.X == self.CC[0][x]).all(axis=1)).any()
        TEST_Y = ((self.Y == self.CC[1][y]).all(axis=1)).any()
        if np.invert(TEST_X) :
            print('OK'); print(self.X)
            self.X[np.random.randint(len(self.X))] = self.CC[0][x]
            print(self.X)
        if np.invert(TEST_Y):
            print('yeah'); print(self.Y)
            self.Y[np.random.randint(len(self.Y))] = self.CC[1][y]
            print(self.Y)
        return self.X, self.Y

    ## Action Exploration/Exploitation Dilemna
    def ACTION(self, Input) :
        img_in = torch.tensor(Input, dtype=torch.float)
        # actor-critic
        action_probs, critic_value = self.MODEL(img_in)
        # exploration-exploitation dilemna
        DILEMNA = np.squeeze(action_probs.detach().numpy())
        if DILEMNA.sum() == 0 or str(DILEMNA.sum()) == 'nan' :
            next_action = np.random.randint(self.IO[1])
        else : 
            p_norm = DILEMNA/DILEMNA.sum()
            next_action = np.random.choice(self.IO[1], p=p_norm)
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
        # actor-critic
        actor, critic = self.MODEL(old_state)
        # Compute predicted Q-values for each action
        pred_q_values_batch = torch.sum(actor.gather(1, action),dim=1).detach()
        pred_q_values_next, critic  = self.MODEL(new_state)
        # Compute targeted Q-value for action performed
        target_q_values_batch = reward+(1-DONE)*self.GAMMA*torch.max(pred_q_values_next, 1)[0]
        # zero the parameter gradients
        self.MODEL.zero_grad()
        # Compute the loss
        self.loss = self.criterion(pred_q_values_batch,target_q_values_batch)
        # Do backward pass
        self.loss.backward()
        self.optimizer.step()
        # reset memory
        self.MEMORY_ = self.MEMORY
        self.MEMORY = [[],[],[],[],[]]
    
    ## reset object
    def RESET(self):
        GRAPH = self.NET.NEXT_GEN(-1)
        XY_TUPLE = (self.X,self.Y)
        return Q_AGENT(*self.ARG, NET = GRAPH, COOR = XY_TUPLE)
    
    ## mutation
    def MUTATION(self, DENSITY_IO, MUT = None):
        # low variation of child density
        DI,DO = DENSITY_IO[0].copy(), DENSITY_IO[1].copy()
        DI[tuple(map(tuple, (self.X+[2,2]).T))] += 200
        DO[tuple(map(tuple, (self.Y+[1,1]).T))] += 500
        DENSITY = (DI/DI.sum(), DO/DO.sum())
        # mutate graph
        GRAPH = self.NET.NEXT_GEN(MUT)
        XY_TUPLE = self.MUTATION_IO(DENSITY)
        return Q_AGENT(*self.ARG, NET = GRAPH, COOR = XY_TUPLE)
    
    ## control group
    def CONTROL_NETWORK(self):
        """
        For Lyfe problem : not generalized
        """
        # init number of connection per layer
        """NB_H_LAYER = 2"""
        """NB_C_P_LAYER = int(np.sqrt(self.IO[0]) + np.sqrt(self.IO[1]))"""
        # network equivalence
        NET = np.array([[-1, 3, 4, 32, [[2,0],[2,1],[2,2],[2,3]]],
                        [ 1, 4, 9, 10, [[0,0],[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8]]],
                        [ 2, 4, 4, 20, [[1,0],[1,1],[1,2],[1,3]]]])
        # Listing
        LIST_C = np.array([[0,0,0],[0,0,1],[0,0,2],[0,0,3],[0,0,4],[0,0,5],[0,0,6],[0,0,7],[0,0,8],
                          [10,1,0],[10,1,1],[10,1,2],[10,1,3],
                          [20,2,0],[20,2,1],[20,2,2],[20,2,3]])
        return [(9,3), NET.copy(), LIST_C.copy()]

if __name__ == '__main__' :
    ARG = ((9,3),25, 16, 16, 12)
    q = Q_AGENT(*ARG)
    #print(q.NEURON_LIST)
    #Mutate
    DXY = (np.ones((5,5))/25,np.ones((3,3))/9)
    for i in range(10) :
        print(i)
        p = q.MUTATION(DXY)
        #print(p.NEURON_LIST)
    plt.imshow(DXY[0])
    