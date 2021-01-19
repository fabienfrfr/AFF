#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:19:36 2021

@author: fabien
"""

import numpy as np

from GRAPH_GEN import GRAPH

TEST_CLASS = True

################################ GRAPH Evolution Augmenting Topology
class GRAPH_EAT(GRAPH):
    def __init__(self, GEN_PARAM, NET):
        if GEN_PARAM != None :
            # first generation
            NB_P_GEN, I, O, P_MIN = GEN_PARAM
            #Inheritance of Graph_gen
            super().__init__(NB_P_GEN, I, O, P_MIN)
        else : 
            self.IO, self.NEURON_LIST, self.LIST_C = NET
        
    def NEXT_GEN(self):
        # copy of module (heritage)
        IO, NEURON_LIST, LIST_C = self.IO, self.NEURON_LIST, self.LIST_C
        # adding mutation (variation)
        MUT = 1 # np.random.randint(2) # 0 : add neuron, 1 : add connect
        if MUT == 0 :
            # add connection (0.8 proba)
            NEURON_LIST = self.ADD_CONNECTION(NEURON_LIST, LIST_C)
        elif MUT == 1 :
            # add neuron (0.1 proba)
            NEURON_LIST, LIST_C = self.ADD_NEURON(NEURON_LIST)
        elif MUT == 2 :
            # add layers (0.1 proba)
            NEURON_LIST, LIST_C = self.ADD_LAYERS(NEURON_LIST, LIST_C)
        elif MUT == 3 :
            # cut doublon connect neuron (0.1 proba)
            NEURON_LIST = self.CUT_CONNECTION(NEURON_LIST)
        elif MUT == 4 :
            # cut neuron
            NEURON_LIST, LIST_C = self.CUT_NEURON(NEURON_LIST)
        # return neuronList with mutation or not
        return GRAPH_EAT(None,[IO, NEURON_LIST, LIST_C])
    
    def ADD_CONNECTION(self, NEURON_LIST, LIST_C) :
        # add Nb connect
        idx = np.random.randint(NEURON_LIST.shape[0])
        NEURON_LIST[idx,2] += 1
        # add element list
        idx_ = np.random.randint(LIST_C.shape[0])
        NEURON_LIST[idx,-1] += [LIST_C[idx_, 1:].tolist()]
        return  NEURON_LIST

    def ADD_NEURON(self, NEURON_LIST):
        NB_LAYER = NEURON_LIST[:,0].max()
        # add neuron in one layers
        if NB_LAYER == 1 : IDX_N = 1
        else : IDX_N = np.random.randint(1,NB_LAYER)
        NEURON_LIST[IDX_N,1] += 1
        idx_new, idx_c_new = NEURON_LIST[IDX_N,0], NEURON_LIST[IDX_N,1]-1
        # add connection
        IDX_C = np.random.randint(NB_LAYER)
        NEURON_LIST[IDX_C, 2] += 1
        NEURON_LIST[IDX_C,-1] += [[idx_new, idx_c_new]]
        # update list_connection
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C

    def ADD_LAYERS(self, NEURON_LIST, LIST_C):
        # new one neuron layers
        idx_new = NEURON_LIST[:,0].max() + 1
        POS_X_new = np.random.randint(1,NEURON_LIST[:,3].max())
        NEW_NEURON = np.array([idx_new, 1, 1, POS_X_new, []])
        # connection of new neuron input
        IDX_C = np.where(LIST_C[:,0] < POS_X_new)[0]
        idx_c = IDX_C[np.random.randint(IDX_C.shape[0])]
        NEW_NEURON[-1] = LIST_C[idx_c, 1:]
        # adding connection of downstream neuron
        IDX_N = np.where(NEURON_LIST[:,3] > POS_X_new)[0]
        idx_n = IDX_N[np.random.randint(IDX_N.shape[0])]
        NEURON_LIST[idx_n, 2] += 1
        NEURON_LIST[idx_n, -1] += [idx_new, 0]
        # add layers and update list
        NEURON_LIST = np.concatenate((NEURON_LIST,NEW_NEURON[None]), axis=0)
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C
    
    def CUT_CONNECTION(self, NEURON_LIST):
        # listing of connection
        CONNECT_DATA = self.CONNECTED_DATA(NEURON_LIST)
        # choose connect duplicate (doublon : ! min connect, otherwise : return)
        c_u, ret = np.unique(CONNECT_DATA[:,2:], axis=0, return_counts=True)
        idx_doublon = np.where(ret > 1)[0]
        if idx_doublon.shape == (0,) :
            return NEURON_LIST
        c_2_cut = c_u[idx_doublon[np.random.randint(len(idx_doublon))]]
        # find cut connection (! 1st link : vestigial, otherwise : return)
        IDX_CD = np.where((CONNECT_DATA[:,1] !=0)*(CONNECT_DATA[:,2:] == c_2_cut).all(axis=1))[0]
        if IDX_CD.shape == (0,) :
            return NEURON_LIST
        idx_cd = IDX_CD[np.random.randint(IDX_CD.shape)][0]
        idx, idx_ = CONNECT_DATA[idx_cd, :2]
        # update neuronlist
        IDX = np.where(NEURON_LIST[:,0] == idx)[0]
        NEURON_LIST[IDX,2] -= 1
        del(NEURON_LIST[IDX,-1][0][int(idx_)])
        return NEURON_LIST
    
    def CUT_NEURON(self, NEURON_LIST):
        # listing of connection
        CONNECT_DATA = self.CONNECTED_DATA(NEURON_LIST)
        ## find possible neuron (no ones connection)
        c_n, ret = np.unique(CONNECT_DATA[:,0], return_counts=True)
        idx_ones = c_n[np.where(ret == 1)[0]]
        # ones verif
        if idx_ones.shape != (0,) :
            bool_o  = np.any([CONNECT_DATA[:,0] == i for i in idx_ones], axis = 0)
            C = CONNECT_DATA[bool_o,2:]
            bool_o_ = np.any([(CONNECT_DATA[:,2:] == d).all(axis=1) for d in C], axis=0)
            # choose neuron
            C_ = CONNECT_DATA[np.invert(bool_o_), 2:]
        else :
            # choose neuron
            C_ = CONNECT_DATA[:, 2:]
        C_ = C_[C_[:,0] != 0] # del input
        idx, idx_ = C_[np.random.randint(C_.shape[0])]
        IDX = np.where(NEURON_LIST[:,0] == idx)[0]
        # remove one neuron number
        NEURON_LIST[IDX, 1] -= 1
        # update list of neuron
        for n in NEURON_LIST :
            list_c = np.array(n[-1])
            # boolean element
            egal = (list_c == [idx, idx_]).all(axis=1)
            sup_ = (list_c[:,0] == idx) * (list_c[:,1] > idx_)
            # change connection (permutation)
            if egal.any() :
                list_c[sup_,1] -= 1
                list_c = list_c[np.invert(egal)]
            # update neuronlist
            n[-1] = list_c.tolist()
            n[2] = len(n[-1])
        # update connection list
        LIST_C = self.LISTING_CONNECTION(NEURON_LIST.shape[0]-1, NEURON_LIST[:,:-1])
        return NEURON_LIST, LIST_C
    
    def CONNECTED_DATA(self, NEURON_LIST):
        CONNECT_DATA = []
        for n in NEURON_LIST :
            idx = n[0]*np.ones((n[2],1))
            idx_ = np.arange(n[2])[:,None]
            c = np.array(n[-1])
            CONNECT_DATA += [np.concatenate((idx,idx_,c),axis=1)]
        CONNECT_DATA = np.concatenate(CONNECT_DATA)
        return CONNECT_DATA

################################ GRAPH TESTER
if TEST_CLASS :
    # Generator
    NB_P_GEN = 16
    P_MIN = 1
    
    # I/O
    I = 3 # image cells
    O = 3 # action
    
    # Init
    NET = GRAPH_EAT([NB_P_GEN, I, O, P_MIN], None)
    print("Liste des neurons : \n", NET.NEURON_LIST)
    
    NEURON_LIST = NET.NEURON_LIST
    
    NET_ = NET.NEXT_GEN()
    print("Liste des neurons : \n", NET_.NEURON_LIST)
        
