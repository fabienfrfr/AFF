#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:01:37 2021
@author: fabien
"""

import torch, torch.nn as nn
import numpy as np
#import torch.nn.functional as F

################################ Custom neural network
class pRNN(nn.Module):
    def __init__(self, NET,B,I, STACK=False):
        super().__init__()
        self.NET = NET
        self.BS = B
        self.STACK = STACK # mini-batch reccurence
        # list of layers
        self.Layers = nn.ModuleList( [nn.Linear(self.NET[0][2], self.NET[0][1])] +
                                     [nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.NET[1:]] +
                                     [nn.Sequential(nn.Conv1d(I, I, 1, groups=I, bias=True), nn.ReLU())])
        # trace data
        self.trace = [torch.zeros(B,n[1]) for n in self.NET] + [torch.zeros(B,I)] # with autogradient
        # pseudo RNN (virtual input)
        self.h = [torch.zeros(B,n[1]) for n in self.NET] + [torch.zeros(B,I)]
    
    def graph2net(self, BATCH_, requires_stack = False):
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
                if requires_stack : tensor[-1] = tensor[-1][None]
            tensor_in = torch.cat(tensor, dim=1)
            self.trace[i][BATCH_] = self.Layers[i](tensor_in)
        return i
    
    def forward(self,x):
        s = x.shape
        # Generalization of Exploitation or Training batch
        BATCH_ = np.arange(len(x))
        # input functionalization (with spread sparsing)
        self.trace[-1][:] = self.Layers[-1](x.view(s[0],s[1],1)).view(s)
        if self.STACK :
            # full adapted but really slow (python & no tensor calc avantage)
            for b in BATCH_ :
                idx_end = self.graph2net(b, requires_stack = True)
                # save t+1 (and periodic bound)
                for t in range(len(self.trace)):
                    if b < self.BS-1 :
                        self.h[t][b+1] = self.trace[t][b].detach()
                    else :
                        self.h[t][0] = self.trace[t][b].detach()
        else :
            # Only adapted for SGD, if mini-batch, pseudo-rnn perturbation
            idx_end = self.graph2net(BATCH_)
            # save for t+1
            for t in range(len(self.trace)):
                self.h[t][BATCH_] = self.trace[t][BATCH_].detach()
        # output probs
        return self.trace[idx_end]

if __name__ == '__main__' :
    IO = (17,3)
    BATCH = 16
    # graph part
    from GRAPH_EAT import GRAPH_EAT
    NET = GRAPH_EAT([IO, 1], None)
    # networks
    model = pRNN(NET.NEURON_LIST, BATCH, IO[0], STACK=False)
    # data test
    tensor_in = torch.randn(BATCH,IO[0])
    tensor_out = model(tensor_in)
    # print
    print(NET.NEURON_LIST,'\n' ,tensor_out,'\n',model.h[0])
    # change network type
    model = pRNN(NET.NEURON_LIST, BATCH, IO[0], STACK=True)
    tensor_out = model(tensor_in)
    print(tensor_out,model.h[0])
