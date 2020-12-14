#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 14:01:21 2020
@author: fabien
"""

import numpy as np

################################ GRAPH of Network
class GRAPH():
    def __init__(self, I,O, NB_P_GEN, Net):
        if Net == [] :
            ### First Net : 'Index','NbNeuron','NbConnect','x','y','listInput'
            C = np.random.randint(1,NB_P_GEN+1)
            self.Net = np.array([[-1, O, C,   2., .5, []],
                                 [ 0, I, I,  -1., .5, []]])
            # Hidden parameter
            R = np.rint(np.sqrt(NB_P_GEN)).astype(int)
            N = np.random.randint(1,R+1)
            # Hidden gen
            Hidden = np.zeros((N, 6), dtype=object)
            Hidden[:,0] = np.arange(N, dtype=int) + 1
            Hidden[:,1] = np.random.randint(1,R-N+2,N, dtype=int)
            Hidden[:,2] = np.random.randint(1,R-N+2,N, dtype=int)
            Hidden[:,3] = np.random.random(N)
            Hidden[:,4] = np.random.random(N)
            Hidden[:,5] = N*[[]]
            # Assembly I/0 and Hidden Layers
            self.Net = np.concatenate((self.Net,Hidden))
            # Construct adjacency
            self.first_connection()
        else :
            self.Net = Net
    
    def first_connection(self):
        for n in self.Net :
            connect = []
            if n[3] == -1 : connect = [[-1,-1]]
            else :
                loc = self.Net[1:,0]
                for i in range(n[2]):
                    idx_ = loc[np.random.randint(loc.shape)]
                    c_out = np.random.randint(int(self.Net[self.Net[:,0] == idx_, 1]))
                    connect += [[int(idx_), int(c_out)]]
            n[-1] = connect
    
    def new_gen(self):
        return GRAPH(None,None,None,self.Net)
    
    def update_net(self):
        alea = 0 #np.random.randint(10)
        if alea == 0 :
            net = np.zeros((1,6), dtype=object)
            net[0,0] = self.Net[:,0].max() + 1
            net[0,1:3] = [1,1]
            net[0,3:5] = np.random.random(2)
            net[0,-1] = [[]]
            self.Net = np.concatenate((self.Net,net))
