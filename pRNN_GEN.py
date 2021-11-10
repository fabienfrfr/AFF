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
        self.trace = (NET.shape[0]+1)*[None]
        # pseudo RNN (virtual input)
        self.h = [torch.zeros(B,n[1]) for n in self.NET] + [torch.zeros(B,I)]
    
    def graph2net(self, BATCH_, return_trace = False):
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
        if return_trace :
            return i, self.trace
        else :
            return i
    
    def forward(self,x):
        s = x.shape
        # Generalization of Exploitation or Training batch
        BATCH_ = np.arange(len(x))
        # input functionalization (with spread sparsing)
        self.trace[-1] = self.Layers[-1](x.view(s[0],s[1],1)).view(s)
        if self.STACK :
            trace = []
            """
            to debug
            """
            # full adapted but really slow (python & no tensor calc avantage)
            for b in BATCH_ :
                idx_end, trace_ = self.graph2net(b, return_trace = True)
                trace += [trace_]
                # save t+1 (and periodic bound)
                for t in range(len(trace_)):
                    if b < self.BS :
                        self.h[t][b+1] = trace_[t][b].detach()
                    else :
                        self.h[t][0] = trace_[t][b].detach()
            # recontruct complete trace
            for t in range(len(trace_)):
                self.trace[t] = torch.cat([trace[i][t] for i in range(len(trace))])
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
    print(tensor_out)