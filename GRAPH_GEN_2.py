#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:01:21 2020
@author: fabien
"""
import numpy as np

# nombre de génération
NB_P_GEN = 16

# nombre de perceptron dans les couches "hidden"
P_MAX = np.rint(np.sqrt(NB_P_GEN)+2).astype(int) # empiric
NB_PERCEPTRON_HIDDEN = np.random.randint(0,P_MAX+1)

# I/O
I = 3 #nombre de case de l'image
O = 3 # nombre d'action

# nombre de connection minimal (invariant)
C_MIN = NB_PERCEPTRON_HIDDEN + I

# nombre de connection maximal (:x)
C_MAX = 2*C_MIN # approx

# nombre de layer dans la couche hidden
if NB_PERCEPTRON_HIDDEN == 0 :
    NB_LAYERS = 0
elif NB_PERCEPTRON_HIDDEN == 1 :
    NB_LAYERS = 1
else :
    NB_LAYERS = np.random.randint(1, NB_PERCEPTRON_HIDDEN + 1)

# nombre de connection per generation
NB_CONNEC_TOT = np.random.randint(C_MIN,C_MAX)
print("Nombre de connection, perceptron et couche : \n", NB_CONNEC_TOT, NB_PERCEPTRON_HIDDEN, NB_LAYERS)

## nb neuron and connection by hidden layer
c = np.random.randint(1,NB_CONNEC_TOT-NB_LAYERS+1)
NEURON_LIST = [[-1, O, c]] # out layers
SUM_N, SUM_C, REMAIN_L = 0, c, NB_LAYERS-1
if NB_LAYERS > 0 :
    for i in range(1,NB_LAYERS+1):
        # recurrent law
        NMAX = NB_PERCEPTRON_HIDDEN - SUM_N - REMAIN_L
        CMAX = NB_CONNEC_TOT - SUM_C - REMAIN_L
        # define number of perceptron per layers 
        if i == NB_LAYERS : 
            n = NMAX
            c = CMAX
        else : 
            n = np.random.randint(1, NMAX+1)
            c = np.random.randint(1, CMAX+1)
        NEURON_LIST += [[i, n, c]]
        # update weight
        SUM_N += n
        SUM_C += c
        REMAIN_L -= 1

NEURON_LIST = np.array(NEURON_LIST)
print("Liste des neuron (idx, neuron, connect) : \n", NEURON_LIST)

# define x position of neuron (Y : it's for visualisation)
MAX_HIDDEN_LVL = 32
X_POS = np.zeros(NB_LAYERS+1, dtype=int)
X_POS[0]  = MAX_HIDDEN_LVL
X_POS[1:] = np.random.randint(1, MAX_HIDDEN_LVL, NB_LAYERS)

NEURON_LIST = np.concatenate((NEURON_LIST,X_POS[None].T), axis=1)
print("Liste des neuron + position : \n", NEURON_LIST)

# listing of possible connection (<= NB_CONNEC_TOT)
LIST_C = [[0,0,i] for i in range(I)] # X, IDX, NEURON
if NB_LAYERS > 0 :
    for l in NEURON_LIST[1:]:
        LIST_C += [[l[-1],l[0], i] for i in range(l[1])]
LIST_C = np.array(LIST_C)
print("Liste des connections : \n", LIST_C)

# redistribution of connection per layer (neirest and normal random (x))
C_PER_LAYERS = []
LIST_C_REMAIN = LIST_C.copy()
for n in NEURON_LIST :
    C_LAYER = []
    pos = n[-1]
    # calculate connection
    for c in range(n[2]) :
        # pos - x
        if LIST_C_REMAIN.shape[0] != 0 :
            d = pos - LIST_C_REMAIN[:,0] 
        else :
            d = pos - LIST_C[:,0]
        # init position probability
        p = np.zeros(d.shape)
        # position
        try : dmin = d[d>0].min()
        except : dmin = 2*MAX_HIDDEN_LVL
        p_behind = d == dmin
        p_upstream = d > 0
        p_front = d <= 0
        # discrete probability low gen
        if c == 0 :
            p[p_behind] = 1.
        else :
            # 1/3 + 2/3
            p[p_front] = 1.
            p[p_upstream] = 2.*((p_front.sum()+1)/(p_upstream.sum()+1))
            p[p_behind] = 2*p[p_behind]
        # normalisation
        p = p/np.sum(p)
        # random connection
        idx = np.random.choice(d.shape[0], 1, p=p)
        if LIST_C_REMAIN.shape[0] != 0 :
            C_LAYER += LIST_C_REMAIN[idx,1:].tolist()
            # remove element in list
            LIST_C_REMAIN = np.delete(LIST_C_REMAIN, idx, 0)
        else :
            C_LAYER += LIST_C[idx,1:].tolist()
    # list of list
    C_PER_LAYERS += [C_LAYER]
if NB_LAYERS > 0 :
    C_PER_LAYERS = np.array(C_PER_LAYERS)
print("Liste des connections par couche : \n", C_PER_LAYERS)

# complete data net construction
if NB_LAYERS > 0 :
    NEURON_LIST = np.concatenate((NEURON_LIST,C_PER_LAYERS[:,None]), axis = 1)
else :
    NEURON_LIST = np.concatenate((NEURON_LIST,C_PER_LAYERS[0]), axis = 1)
print("Liste des neuron + connectList : \n", NEURON_LIST)
