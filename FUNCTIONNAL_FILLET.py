#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:09:28 2021
@author: fabien
"""

# ML module
import numpy as np, pylab as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.utils import shuffle

from tqdm import tqdm

# networks construction
from GRAPH_EAT import GRAPH_EAT
from pRNN_GEN import pRNN

# calculation (multi-cpu) and data (namedtuple) optimisation
#import multiprocessing, collections

# control net
class CTRL_NET(nn.Module):
    def __init__(self, IO):
        super(CTRL_NET, self).__init__()
        I,O = IO
        H = int(np.sqrt(I))
        self.IN = nn.Conv1d(I, I, 1, groups=I, bias=True)
        self.H1 = nn.Linear(I, H)
        self.H2 = nn.Linear(H, H)
        self.OUT = nn.Linear(H, O)

    def forward(self, x):
        s = x.shape
        x = F.relu(self.IN(x.view(s[0],s[1],1)).view(s))
        x = F.relu(self.H1(x))
        x = F.relu(self.H2(x))
        return self.OUT(x)

# ff module
class model():
    def __init__(self, IO, SAMPLE_SIZE, BATCH_SIZE, EPOCH, NB_GEN, NB_SEEDER, DATA_XPLT = 0.01, LEARNING_RATE = 1e-6, MOMENTUM = 0.5):
        # Parameter
        self.IO = IO
        self.N = SAMPLE_SIZE
        self.BATCH = BATCH_SIZE
        self.EPOCH = EPOCH
        self.NB_GEN = NB_GEN
        self.NB_SEEDER = NB_SEEDER**2
        self.LR = LEARNING_RATE
        self.MM = MOMENTUM
        # generate first ENN step
        self.GRAPH_LIST = [GRAPH_EAT([self.IO, 1], None) for n in range(self.NB_SEEDER-1)]
        self.SEEDER_LIST = [CTRL_NET(self.IO)]
        for g in self.GRAPH_LIST :
            NEURON_LIST = g.NEURON_LIST
            self.SEEDER_LIST += [pRNN(NEURON_LIST, self.BATCH, self.IO[0])]
        # generate loss-optimizer
        self.OPTIM = [torch.optim.SGD(s.parameters(), lr=LEARNING_RATE,momentum=MOMENTUM) for s in self.SEEDER_LIST]
        self.CRITERION = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
        self.LOSS = self.NB_SEEDER*[0]
        # calculate nb batch per generation
        self.NB_BATCH_P_GEN = int((DATA_XPLT*self.N*self.EPOCH)/(self.NB_GEN*self.BATCH))
        # selection and accuracy
        self.SCORE = []
        self.ACCUR = [] # in %
        # for next gen (n-plicat) and control group
        self.NB_CONTROL = 1 # always (preference)
        self.NB_CHALLENGE = int(np.sqrt(self.NB_SEEDER)-self.NB_CONTROL)
        self.NB_SURVIVOR = self.NB_CHALLENGE # square completion
        self.NB_CHILD = int(np.sqrt(self.NB_SEEDER)-1) # FITNESS
    
    def fit(self, DATA, LABEL):
        # gen loop
        for _o in tqdm(range(self.NB_GEN)):
            DATA,LABEL = shuffle(DATA,LABEL)
            # compilation
            for n in range(self.NB_BATCH_P_GEN):
                data = torch.tensor(DATA[n*self.BATCH:(n+1)*self.BATCH].reshape(-1,self.IO[0]), dtype=torch.float)
                target = torch.tensor(LABEL[n*self.BATCH:(n+1)*self.BATCH]).type(torch.LongTensor)
                # seed
                for s in range(self.NB_SEEDER):
                    self.OPTIM[s].zero_grad()
                    output = self.SEEDER_LIST[s](data)
                    self.LOSS[s] = self.CRITERION[s](output,target)
                    self.LOSS[s].backward()
                    self.OPTIM[s].step()
                # score loss
                self.SCORE += [torch.tensor(self.LOSS).numpy()[None]]
            # score accuracy
            dt_train = torch.tensor(DATA[np.random.randint(5)])
            # self.predict(dt_train) 
            self.ACCUR += []
            # evolution
            SCORE_LIST = self.SCORE[-1].squeeze()
            ## fitness (in accuracy test)
            ORDER = np.argsort(SCORE_LIST[self.NB_CONTROL:]).astype(int)
            # control and survivor
            CTRL = self.SEEDER_LIST[:self.NB_CONTROL]
            BEST = [self.SEEDER_LIST[self.NB_CONTROL:][n] for n in ORDER[:self.NB_SURVIVOR]]
            B_G_ = [self.GRAPH_LIST[n] for n in ORDER[:self.NB_SURVIVOR]]
            # challenger
            NEWS = []
            N_G_ = []
            for n in range(self.NB_CHALLENGE) :
                N_G_ += [GRAPH_EAT([self.IO, 1], None)]
                NEWS += [pRNN(N_G_[-1].NEURON_LIST, self.BATCH, self.IO[0])]
            # mutation
            MUTS = []
            M_G_ = []
            for g in B_G_ :
                for i in range(self.NB_CHILD):
                    M_G_ += [g.NEXT_GEN()]
                    MUTS += [pRNN(M_G_[-1].NEURON_LIST, self.BATCH, self.IO[0])]
            # update
            self.SEEDER_LIST = CTRL + BEST + MUTS + NEWS
            self.GRAPH_LIST = B_G_ + M_G_ + N_G_
            # generate loss-optimizer
            self.OPTIM = [torch.optim.SGD(s.parameters(), lr=self.LR,momentum=self.MM) for s in self.SEEDER_LIST]
            self.CRITERION = [nn.CrossEntropyLoss() for n in range(self.NB_SEEDER)]
        # compact data
        self.SCORE = np.concatenate(self.SCORE).T
        
    def predict(self, DATA, WITH_VAL = True):
        N = DATA.shape[0]
        with torch.no_grad():
            log_prob = []
            for s in self.SEEDER_LIST :
                log_prob += [s(DATA)]
        proba = torch.exp(torch.cat(log_prob).reshape(N,-1,10))
        max_values, max_index = torch.max(proba,2)
        if WITH_VAL :
            return max_values, max_index
        else : 
            return max_index

### TESTING PART
if __name__ == '__main__' :
    # module for test
    from tensorflow.keras.datasets import mnist
    # import mnist
    (X_train_data,Y_train_data),(X_test_data,Y_test_data) = mnist.load_data()
    # shuffle data
    X_train_data,Y_train_data = shuffle(X_train_data,Y_train_data)
    plt.imshow(X_train_data[0]); print(Y_train_data[0]); plt.show(); plt.close()
    # data resizing
    X_train_data, X_test_data = X_train_data[:,::2,::2], X_test_data[:,::2,::2]
    plt.imshow(X_train_data[0]); plt.show(); plt.close()
    # data info
    N, x, y = X_train_data.shape ; I = x*y
    O = np.unique(Y_train_data).size
    # parameter
    BATCH, EPOCH = 20, 2
    NB_GEN, NB_SEED = 20, 5
    # init
    MODEL = model((I,O), N, BATCH, EPOCH, NB_GEN,NB_SEED)
    # training
    MODEL.fit(X_train_data,Y_train_data)
    plt.matshow(MODEL.SCORE, aspect="auto"); plt.show(); plt.close()
    # predict
    plt.imshow(X_test_data[0]); plt.show(); plt.close()
    X_torch = torch.tensor(X_test_data[:3].reshape((-1,x*y)), dtype=torch.float)
    X_torch = torch.tensor(X_test_data[0].reshape((-1,x*y)), dtype=torch.float)
    max_v, max_i = MODEL.predict(X_torch)
    print("Predicted Digit =", max_i)
