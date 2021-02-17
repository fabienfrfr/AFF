#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 13:42:20 2021
@author: fabien
"""

import torch, torch.nn as nn
import torch.optim as optim
import numpy as np

from Q_AGENT_GEN import Q_AGENT

import pylab as plt

### PARAMETER PART
TEST = True

### FUNCTION PART
def DRAW_NETWORK(net,in_):
    ## Generate layer node
    # TRIPLET : (IDX, X, INDEX_ = Y)
    neuron_in, neuron_out = [], []
    # Input part :
    for n in range(in_):
        neuron_out += [[0,0,n]]
    # Layering
    for n in net :
        # input part
        for n_ in range(n[2]) :
            neuron_in  += [[n[0],n[3]-0.25,n_]]
        # output part
        for n_ in range(n[1]) :
            neuron_out  += [[n[0],n[3]+0.25,n_]]
    neuron_in, neuron_out = np.array(neuron_in), np.array(neuron_out)
    ## Connect each Node
    for n in net :
        i = 0
        for n_ in n[-1] :
            connect_a = [n[3]-0.25, i]
            idx = np.where((n_ == neuron_out[:, 0::2]).all(axis=1))[0]
            connect_b = neuron_out[idx][:,1:]
            X = np.concatenate((np.array([connect_a]), connect_b))
            plt.plot(X[:,0], X[:,1], 'k')
            # increment
            i+=1
    ## Plot the graph-network
    plt.scatter(neuron_in[:,1], neuron_in[:,2], s=10); plt.scatter(neuron_out[:,1], neuron_out[:,2], s=30)
    
### TESTING PART
if TEST :
    # Parameter
    IO = (9,3)
    NB_P_GEN = 16
    batch_size = 32
    MAP_SIZE = 16
    ## Init
    AGENT = Q_AGENT(NB_P_GEN, IO, batch_size)
    
    ############### plot neuron list
    NET = AGENT.NEURON_LIST
    DRAW_NETWORK(NET,IO[0])
