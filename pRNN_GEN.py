#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:01:21 2020
@author: fabien
"""
import torch, torch.nn as nn
import numpy as np, pylab as plt

################################ PARAMETER
I,O = 5,5
NB_NEURON = 2
NX, NY = 5, 5
batch_size = 10
# Out connectivity
C = 5

################################ First GEN
### First Net : 'Index','NbNeuron','NbConnect','x','y','listInput'
Net = np.array([[-1, O, C,  NX, NY/2, []],
                [ 0, I, I,  -1, NY/2, []]])

# Hidden part
Hidden = np.zeros((NB_NEURON, 6), dtype=object)

Hidden[:,0] = np.arange(NB_NEURON) + 1
Hidden[:,1] = np.ones(NB_NEURON, dtype=int)
Hidden[:,2] = np.ones(NB_NEURON, dtype=int)
Hidden[:,3] = np.random.randint(0, NX, NB_NEURON)
Hidden[:,4] = np.random.randint(0, NY, NB_NEURON)
Hidden[:,5] = NB_NEURON*[[]]

# Concatenate
Net = np.concatenate((Net,Hidden))

#### Adjacency in Net
for n in Net :
    connect = []
    if n[3] == -1 : connect = [[-1,-1]]
    else :
        loc = Net[1:,0] #[Net[:,3] < n[3]][:,0]
        for i in range(n[2]):
            idx_ = loc[np.random.randint(loc.shape)]
            c_out = np.random.randint(int(Net[Net[:,0] == idx_, 2]))
            connect += [[idx_, c_out]]
    n[-1] = connect

################################ Custom neural network
class Network(nn.Module):
    def __init__(self, Net, batch_size):
        super().__init__()
        self.Net = Net
        # list of layers
        self.Layers = nn.ModuleList([nn.Sequential(nn.Linear(n[2], n[1]), nn.ReLU()) for n in self.Net])
        self.trace = [torch.zeros(batch_size,n[2]).requires_grad_() for n in self.Net]
        # virtual trace (pseudo-rnn)
        self.h = None

    def forward(self,x):
        # virtualization (t-1)
        self.h = [t.detach() for t in self.trace]
        # BP follow XY position
        order = np.argsort(self.Net[:, 3])
        for i in range(self.Net.shape[0]) :
            idx = order[i]
            if i == 0 : x = self.Layers[idx](x)
            else :
                tensor = []
                for j,k in self.Net[idx, -1] :
                    idx_ = np.where(Net[:,0] == j)[0][0]
                    # pseudo-RNN
                    if (self.Net[idx_, 3] >= self.Net[idx, 3]) : tensor += [self.h[idx_][:,None,k]]
                    # Non Linear input
                    else : tensor += [self.trace[idx_][:,None,k]]
                tensor_in = torch.cat(tensor, dim=1)
                x = self.Layers[idx](tensor_in)
            self.trace[idx] = x
        return x

##### Plot network
fig = plt.figure()
ax = fig.add_subplot()
plt.xlim(-1,NX); plt.ylim(0,NY)
for n in Net :
    for a in n[-1] :
        idx = np.where(a[0] == Net[:,0])[0][0]
        x,y = [n[3],Net[idx,3]], [n[4],Net[idx,4]]
        ax.plot(x,y, 'o-')

##### XOR Fit Part
X = np.mgrid[0:batch_size,0:I][1]
y = 1*np.logical_xor(X < 3, X > 5)

# Convert to tensor
X, y = torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# init RNN
h0 = torch.zeros(batch_size,1).requires_grad_()

# Model init
model = Network(Net, batch_size)
criterion = torch.nn.MSELoss(reduction='sum') # MSE for regression
#criterion = nn.CrossEntropyLoss() # CE for classification (need softmax)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)
# Training Loop
for t in range(1000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(X)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()#retain_graph=True)
    optimizer.step()

##### XOR Predict Part
y_pred = model(X)
print(Net,y,y_pred)
