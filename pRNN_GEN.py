#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:01:21 2020
@author: fabien
"""

import torch, torch.nn as nn
import numpy as np

################################ Custom neural network
class pRNN(nn.Module):
    def __init__(self, Net, batch_size):
        super().__init__()
        self.Net = Net
        # list of layers
        self.Layers = nn.ModuleList([nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.Net])
        self.trace = [torch.zeros(batch_size,n[1]) for n in self.Net]
        # virtual trace (pseudo-rnn)
        self.h = None

    def forward(self,x):
        # virtualization (t-1)
        self.h = [t.detach() for t in self.trace]
        # BP follow XY position
        order = np.argsort(self.Net[:, 3])
        for i in range(self.Net.shape[0]) :
            idx = order[i]
            # input
            if i == 0 : x = self.Layers[idx](x)
            # hidden + output
            else :
                tensor = []
                for j,k in self.Net[idx, -1] :
                    idx_ = np.where(self.Net[:,0] == j)[0][0]
                    # pseudo-RNN
                    if (self.Net[idx_, 3] >= self.Net[idx, 3]) : tensor += [self.h[idx_][:,None,k]]
                    # Non Linear input
                    else : tensor += [self.trace[idx_][:,None,k]]
                tensor_in = torch.cat(tensor, dim=1)
                x = self.Layers[idx](tensor_in)
            self.trace[idx] = x
        return x
