#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:01:21 2020
@author: fabien
"""

import torch, torch.nn as nn
import numpy as np

TEST_CLASS = False

################################ Custom neural network
class pRNN(nn.Module):
    def __init__(self, Net, batch_size):
        super().__init__()
        self.Net = Net
        # list of layers
        self.Layers = nn.ModuleList([nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.Net])
        self.trace = [torch.zeros(batch_size,n[1]) for n in self.Net]
        # input memory
        self.input = None
        # virtual trace (pseudo-rnn)
        self.h = None

    def forward(self,x):
        self.input = x
        # Generalization of Exploitation or Training batch
        BATCH_ = np.arange(len(x)) # min = 1, max = batch_size 
        # virtualization (t-1)
        self.h = [t.detach() for t in self.trace]
        # BP follow XY position
        order = np.argsort(self.Net[:, 3])
        for i in range(self.Net.shape[0]) :
            idx = order[i]
            tensor = []
            for j,k in self.Net[idx, -1] :
                # input
                if j == 0 : tensor += [self.input[:,None,k]]
                # hidden
                else :
                    idx_ = np.where(self.Net[:,0] == j)[0][0]
                    # pseudo-RNN
                    if (self.Net[idx_, 3] >= self.Net[idx, 3]) : tensor += [self.h[idx_][BATCH_,None,k]]
                    # Non Linear input
                    else : tensor += [self.trace[idx_][BATCH_,None,k]]
            tensor_in = torch.cat(tensor, dim=1)
            x = self.Layers[idx](tensor_in)
            self.trace[idx][BATCH_] = x
        return x

################################ GRAPH TESTER
if TEST_CLASS :
    from GRAPH_GEN import GRAPH
    # Generator
    NB_P_GEN = 16
    P_MIN = 1
    
    # I/O
    I = 16 # image cells
    O = 16 # action
    
    NET = GRAPH(NB_P_GEN, I, O, P_MIN)
    Net = NET.NEURON_LIST
    print("Liste des neurons : \n", Net)
    
    batch_size = 5
    # IO values
    X = np.mgrid[0:batch_size,0:I][1]
    y = 1*np.logical_xor(X < 3, X > 7)
    X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    print("XOR output (not shuffleled..) : \n", y)
    
    # init RNN
    h0 = torch.zeros(batch_size,1).requires_grad_()

    # Model init
    model = pRNN(Net, batch_size)
    print("Model : \n", model)
    
    # test prediction
    y_pred = model(X)
    print("first output predicted : \n", y_pred)
