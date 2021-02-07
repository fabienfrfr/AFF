#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 11:38:14 2021
@author: fabien
"""
import torch
import numpy as np

from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

TEST_CLASS = True

################################ GRAPH of Network
class Q_AGENT_GEN():
    def __init__(self, NB_P_GEN, IO, BS):
        self.P_MIN = 1
        # I/O
        self.IO = IO # image cells, action
        self.batch_size = BS
        ## Init
        self.NET = GRAPH_EAT([NB_P_GEN, self.IO[0], self.IO[1], self.P_MIN], None)
        self.NEURON_LIST = self.NET.NEURON_LIST
        self.MODEL = pRNN(self.NEURON_LIST, self.batch_size)
        ## IO Coordinate
        self.X,self.Y = self.FIRST_IO_COOR_GEN()
    
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
        
    def IN_COOR_GEN(self) :
        # random 9 cell in 5*5 
        return np.mgrid[-1:2,-1:2]
    
    ## Training Q-Table
    def OPTIM_MODEL(self) :
        return None
    
################################ GRAPH TESTER
if TEST_CLASS :
    import pylab as plt
    # Parameter
    IO = (9,3)
    NB_P_GEN = 16
    batch_size = 5
    MAP_SIZE = 16
    # Basic Env Gen
    IMG = np.random.random((batch_size,MAP_SIZE,MAP_SIZE))
    POS = np.random.randint(0,MAP_SIZE,2)
    # Init
    AGENT = Q_AGENT_GEN(NB_P_GEN, IO, batch_size)
    # Input test
    In_COOR = np.mod(POS + AGENT.X, MAP_SIZE)
    
    Input = IMG[:,In_COOR[:,0],In_COOR[:,1]]
    IN_ = torch.tensor(Input, dtype=torch.float)
    # Test model
    OUT_ = AGENT.MODEL(IN_)
    MVT = int(torch.argmax(OUT_[0]))
    Out_COOR = np.mod(POS + AGENT.Y[2], MAP_SIZE)
    # Plot
    plt.imshow(IMG[0])
    plt.scatter(Out_COOR[0],Out_COOR[1], c='r', s=120)
    plt.scatter(POS[0],POS[1], c='k', s=80)
    plt.scatter(In_COOR[:,0],In_COOR[:,1], c='b',s=40)
    print(OUT_)
