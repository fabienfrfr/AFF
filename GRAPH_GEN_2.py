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
P_MAX = np.rint(np.sqrt(NB_P_GEN)).astype(int) # empiric
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
print(NB_CONNEC_TOT, NB_PERCEPTRON_HIDDEN, NB_LAYERS)

# nb neuron by hidden layer (à faire)
NEURON_LIST = [[-1, O]]
SUM_N, REMAIN_L = 0, NB_LAYERS-1
if NB_LAYERS > 0 :
    for i in range(NB_LAYERS):
        # recurrent law
        NMAX = NB_PERCEPTRON_HIDDEN - SUM_N - REMAIN_L
        # define number of perceptron per layers 
        if NMAX == 1 : n = 1
        elif i == NB_LAYERS - 1 : n = NMAX
        else : n = np.random.randint(1, NMAX)
        NEURON_LIST += [[i, n]]
        # update weight
        SUM_N += n
        REMAIN_L -= 1

print(NEURON_LIST)

# define x position of neuron (y it's for visualisation)
