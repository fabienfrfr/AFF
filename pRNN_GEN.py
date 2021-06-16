#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:01:37 2021
@author: fabien
"""

import torch, torch.nn as nn
import numpy as np

################################ Custom neural network
class pRNN(nn.Module):
    def __init__(self, NET,B,I):
        super().__init__()
        self.NET = NET
        self.BS = B
        # list of layers
        self.Layers = nn.ModuleList( [nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.NET] +
                                     [nn.Sequential(nn.Linear(self.NET[0][2], 1), nn.ReLU())] +
                                     [nn.Sequential(nn.Conv1d(I, I, 1, groups=I, bias=True), nn.ReLU())])
        # trace data
        self.trace = (NET.shape[0]+2)*[None]
        # pseudo RNN (virtual input)
        self.h = [torch.zeros(B,n[1]) for n in self.NET] + [torch.zeros(B,1)] + [torch.zeros(B,I)]
        
    def forward(self,x):
        s = x.shape
        # Generalization of Exploitation or Training batch
        BATCH_ = np.arange(len(x))
        # input functionalization (with spread sparsing)
        self.trace[-1] = self.Layers[-1](x.view(s[0],s[1],1)).view(s)
        # hidden to output (X ordered)
        for i in np.argsort(self.NET[:, 3]) :
            tensor = []
            for j,k in self.NET[i, -1] :
                # input
                if j == 0 : tensor += [self.trace[-1][BATCH_,None,k]]
                # hidden
                else :
                    # pseudo-RNN (virtual input)
                    if (self.NET[i, 3] >= self.NET[i, 3]) : tensor += [self.h[j][BATCH_,None,k]]
                    # Non Linear input
                    else : tensor += [self.trace[j][BATCH_,None,k]]
            tensor_in = torch.cat(tensor, dim=1)
            self.trace[i] = self.Layers[i](tensor_in)
        # critic part
        self.trace[-2] = self.Layers[-2](tensor_in)
        # save for t+1
        for t in range(len(self.trace)):
            self.h[t][BATCH_] = self.trace[t][BATCH_].detach()
        # output Actor, Critic
        return self.trace[i], self.trace[-2]
