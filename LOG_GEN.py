#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 23:05:53 2021
@author: fabien
"""

import numpy as np, pylab as plt
import pandas as pd, os

################################ SAVE EXP INFO & pre-treatment
class LOG_INFO():
    def __init__(self, PL_LIST, ENV_LIST, GEN):
        # exp info
        self.DF_1 = pd.DataFrame(columns=['ID','GEN','TREE','MAP_SIZE','DENSITY_IN','DENSITY_OUT','IN_COOR','OUT_COOR','NEURON_LIST'])
        self.DF_2 = pd.DataFrame(columns=['ID','AGENT_POS','PNJ_POS','TAG_STATE','SCORE', 'RANK'])
        self.DF = self.DF_1 + self.DF_2
        # density
        D_I, D_O = np.ones((5,5)), np.ones((3,3))
        self.DENSITY_IO = D_I/D_I.sum(), D_O/D_O.sum()
        # Init dataframe
        self.START_CYCLE(PL_LIST, ENV_LIST, GEN)
    
    def SAVE_CSV(self, TIME) :
        FOLDER = 'OUT'
        if(not os.path.isdir(FOLDER)): os.makedirs(FOLDER)
        if type(TIME) != type('') :
            TIME = '_'
        self.DF.to_csv(FOLDER + os.path.sep + 'LYFE_EXP_' + TIME + '_.csv', sep=';', index=False)
    
    def START_CYCLE(self, PLAYS_LIST, ENV_LIST, GEN, SLC_LIST = None):
        # Listing
        ID_ = []
        GEN_ = []
        TREE = []
        MAP_SIZE = []
        DENSITY_IN = []
        DENSITY_OUT = []
        IN_COOR = []
        OUT_COOR = []
        NEURON_LIST = []
        # survival, legacy, challenger
        if SLC_LIST != None :
            c, s, l, f = -1, 0, 1, self.DF_1['TREE'].max()[0]+1
        # Loop
        for i in range(len(PLAYS_LIST)) :
            ID_ += [self.DF_1.shape[0]+i]
            GEN_ += [GEN]
            if SLC_LIST == None :
                TREE += [[i]]
            else :
                L = SLC_LIST[i]
                L_ = 0
                # Survival
                if L[0] == 's' :
                    TREE += [[L[1]]+[s]]
                # Legacy
                elif L[0] == 'l' :
                    if L[1] == L_ :
                        TREE += [[L[1]]+[l]]
                        l += 1
                    else :
                        L_ = L[1]
                        l = 1
                        TREE += [[L[1]]+[l]]
                # Challenger
                elif L[0] == 'f' :
                    TREE += [[f]]
                    f += 1
                elif L[0] == 'c' :
                    TREE += [[c]]
            MAP_SIZE += [ENV_LIST[i].MAP_SIZE]
            D_I, D_O = self.DENSITY_IO
            DENSITY_IN += [list(D_I.reshape(-1))]
            DENSITY_OUT += [list(D_O.reshape(-1))]
            IN_COOR += [PLAYS_LIST[i].X.tolist()]
            OUT_COOR += [PLAYS_LIST[i].Y.tolist()]
            NEURON_LIST += [PLAYS_LIST[i].NEURON_LIST.tolist()]
        # Array construction
        ARRAY = np.zeros((len(PLAYS_LIST),self.DF_1.columns.size), dtype=object)
        for i in range(len(PLAYS_LIST)):
            ARRAY[i,0] = ID_[i]
            ARRAY[i,1] = GEN_[i]
            ARRAY[i,2] = TREE[i]
            ARRAY[i,3] = MAP_SIZE[i]
            ARRAY[i,4] = DENSITY_IN[i]
            ARRAY[i,5] = DENSITY_OUT[i]
            ARRAY[i,6] = IN_COOR[i]
            ARRAY[i,7] = OUT_COOR[i]
            ARRAY[i,8] = NEURON_LIST[i]
        # UPDATE DF1
        DF_1_NEW = pd.DataFrame(ARRAY, columns=list(self.DF_1))
        self.DF_1 = self.DF_1.append(DF_1_NEW, ignore_index=True)
    
    def FINISH_CYCLE(self, ENV_LIST, SCORE, RANK):
        # Listing
        ID = np.arange(self.DF_2.shape[0], self.DF_2.shape[0] + len(ENV_LIST))
        AGENT_POS = []
        PNJ_POS = []
        TAG_STATE = []
        # Loop 
        for e in ENV_LIST :
            AGENT_POS += [e.AG_LIST]
            PNJ_POS += [e.PNJ_LIST]
            TAG_STATE += [e.IT_LIST]
        # Array construction
        ARRAY = np.zeros((len(ENV_LIST),self.DF_2.columns.size), dtype=object)
        for i in range(len(ENV_LIST)):
            ARRAY[i,0] = ID[i]
            ARRAY[i,1] = AGENT_POS[i]
            ARRAY[i,2] = PNJ_POS[i]
            ARRAY[i,3] = TAG_STATE[i]
            ARRAY[i,4] = SCORE[i]
            ARRAY[i,5] = RANK[i]
        # UPDATE DF2
        DF_2_NEW = pd.DataFrame(ARRAY, columns=list(self.DF_2))
        self.DF_2 = pd.concat([self.DF_2, DF_2_NEW])
        # MERGE DF1 + DF2 (pointer)
        self.DF = pd.merge(self.DF_1, self.DF_2, on="ID")
    
    def DENSITY(self, PLAYS, ORDER, NB_GEN, IMSHOW=False):
        IN_DENSITY = np.zeros((5,5))
        OUT_DENSITY = np.zeros((3,3))
        # loop
        RANK = np.arange(NB_GEN)[::-1]
        for n in range(NB_GEN) :
            X_ = PLAYS[ORDER[n]].X
            Y_ = PLAYS[ORDER[n]].Y
            X_CENTER, Y_CENTER = [2,2], [1,1]
            ## Listing
            X = X_ + X_CENTER
            Y = Y_ + Y_CENTER
            # update density
            IN_DENSITY[tuple(map(tuple, X.T))] += RANK[n]
            OUT_DENSITY[tuple(map(tuple, Y.T))] += RANK[n]
        # 1er norm
        IN_DENSITY = IN_DENSITY/IN_DENSITY.sum()
        OUT_DENSITY = OUT_DENSITY/OUT_DENSITY.sum()        
        # (t) (t+1) sum
        IN_DENSITY = self.DENSITY_IO[0] + IN_DENSITY
        OUT_DENSITY = self.DENSITY_IO[1] + OUT_DENSITY
        # 2nd norm
        IN_DENSITY  = IN_DENSITY/IN_DENSITY.sum()
        OUT_DENSITY = OUT_DENSITY/OUT_DENSITY.sum()
        # update density
        self.DENSITY_IO = IN_DENSITY, OUT_DENSITY
        if IMSHOW :
            # PLOT (provisoire)
            plt.imshow(IN_DENSITY); plt.colorbar(); plt.show(); plt.close()
            plt.imshow(OUT_DENSITY); plt.colorbar(); plt.show(); plt.close()
     
    # DEPRECIED
    def ROTATION_3(self, X_CENTER, NEW_CENTER) :
        THETA = np.linspace(np.pi/2, (3./2)*np.pi, 3)
        X_ROTATED = []
        for t in THETA :
            # rotation matrix
            r = np.array(( (np.cos(t), -np.sin(t)),(np.sin(t),  np.cos(t)) ))
            # rotation
            X_ROT = r.dot(X_CENTER.T)
            # increment
            X_ROTATED += [np.rint(X_ROT.T).astype(int) + NEW_CENTER]
        return X_ROTATED