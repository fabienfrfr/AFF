#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:09:28 2021
@author: fabien
"""

import numpy as np, pylab as plt

import torch, torch.nn as nn

from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN


class model():
    def __init__(self, IO, BATCH_SIZE, NB_GEN, NB_SEEDER):
        # Parameter
        self.BATCH_SIZE = BATCH_SIZE
        self.NB_GEN = NB_GEN
        self.NB_SEEDER = NB_SEEDER
        # generate first ENN step
        self.GRAPH_LIST = [GRAPH_EAT([NB_SEEDER, IO[0], IO[1], 1], None) for n in range(NB_SEEDER)]
        self.SEEDER_LIST = []
        for g in self.GRAPH_LIST :
            NEURON_LIST = g.NEURON_LIST
            self.SEEDER_LIST += [pRNN(NEURON_LIST, BATCH_SIZE, IO[0])]
        # generate loss-optimizer
        self.LOSS = [nn.MSELoss(reduction='sum') for n in range(NB_SEEDER)]
        self.OPTIM = [torch.optim.SGD(s.parameters(), lr=1e-6) for s in self.SEEDER_LIST]
        
    def fit(self, DATA, LABEL):
        # Store data/label
        self.DATA = DATA
        self.LABEL = LABEL
        # Training Loop :
        for i in range(self.NB_GEN):
            for j in range(self.NB_SEEDER):
                # update seeder list
                print(i,j)
    
    def compile_(self, epochs, train_dl, criterion, optimizer) :
        # iterate through all the epoch
        for epoch in range(epochs):
            # go through all the batches generated by dataloader
            for i, (inputs, targets) in enumerate(train_dl):
                    # clear the gradients
                    optimizer.zero_grad()
                    # compute the model output
                    yhat = model(inputs)
                    # calculate loss
                    loss = criterion(yhat, targets.type(torch.LongTensor))
                    # credit assignment
                    loss.backward()
                    # update model weights
                    optimizer.step()
        # evaluate accuracy_score
        """
        """
        score = 1
        # return score
        return score
    
    def predict(self, DATA_NEW):
        # Put data in best model
        self.LABEL_NEW = self.SEEDER_LIST[-1](DATA_NEW)
        print(self.SEEDER_LIST[-1])
        # return result
        return self.LABEL_NEW

if __name__ == '__main__' :
    # module for test
    from tensorflow.keras.datasets import mnist
    # import mnist
    (X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()
    plt.imshow(X_train_data[0]); plt.show(); plt.close()
    # data info
    N, x, y = X_train_data.shape
    f = np.unique(Y_train_data).size
    # init
    model = model([x*y,f],10,10,10)
    # training
    model.fit(X_train_data,Y_train_data)
    # predict
    X_torch = torch.tensor(X_test_data[0].reshape((-1,))[np.newaxis], dtype=torch.float)
    result = model.predict(X_torch)
