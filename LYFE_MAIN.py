#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:22:45 2021
@author: fabien
"""

import numpy as np

import time, datetime
#import skvideo.io as io

from Q_AGENT import Q_AGENT
from TAG_ENV import TAG_ENV
from LOG_GEN import LOG_INFO

################################ EXPERIMENTAL PARAMETER (inverted name : need 2 change)
IO = (9,3) # don't change here (not optimized yet)
NB_GEN = 10
batch_size = 16
MAP_SIZE = 16
N_TIME = 12
NB_P_GEN = 5**2

ARG_TUPLE = (IO,NB_GEN, batch_size, MAP_SIZE, N_TIME, NB_P_GEN)

################################ LYFE EXPERIMENT's 
class LYFE():
    def __init__(self, arg):
        # Parameter
        self.NB_P_GEN = arg[-1]
        self.MAP_SIZE = arg[3]
        self.ARG = arg
        # Loop generate player and env (change if common of separated)
        self.PLAYERS, self.ENV = [], []
        for n in range(self.NB_P_GEN) :
            self.PLAYERS += [Q_AGENT(*self.ARG[:-1])]
            AGENT_VIEW, AGENT_MOVE = self.PLAYERS[-1].X, self.PLAYERS[-1].Y
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE))]
            self.PLAYERS[-1].INIT_ENV(self.ENV[-1])
        # Classement & party info
        self.SCORE_LIST = []
        self.GEN = 0
        self.TIME = [0,0]
        # Save-info
        self.INFO_LOG = LOG_INFO(self.PLAYERS, self.ENV, self.GEN)
        self.SLC, self.SLC_1, self.SLC_2 = None, None, None
        # for next gen (n-plicat)
        self.NB_CLUSTER = int(np.sqrt(self.NB_P_GEN))
        
    def LAUNCH(self, VIDEO = False):
        self.TIME[0] = time.time()
        ## party game
        for p,e in zip(self.PLAYERS, self.ENV) :
            p.PARTY(e)
            self.SCORE_LIST += [e.SCORE]
            print(self.SCORE_LIST[-1])
        ## Pre-analysis
        ORDER = np.argsort(self.SCORE_LIST)[::-1]
        # complete cycle
        self.INFO_LOG.FINISH_CYCLE(self.ENV, self.SCORE_LIST, ORDER[::-1])
        # density (after cycle)
        self.INFO_LOG.DENSITY(self.PLAYERS, ORDER, self.NB_P_GEN)
        # update gen
        self.GEN += 1
        # time exp
        self.TIME[1] = time.time()
        print(self.TIME[1] - self.TIME[0])
        #### NEXT GEN
        BEST = ORDER[:self.NB_CLUSTER-1]
        # SURVIVOR
        OLD_PLAYER = self.SURVIVOR(BEST)
        # MUTATION
        MUT_PLAYER = self.LEGACY(OLD_PLAYER)
        # CHALLENGER
        NEW_PLAYER = self.FOLLOWER()
        # UPDATE
        self.PLAYERS = OLD_PLAYER + MUT_PLAYER + NEW_PLAYER
        self.ENV_UPDATE()
        ## Re-init cycle
        SLC_1 = [len(OLD_PLAYER)*['s']+len(MUT_PLAYER)*['l']+len(NEW_PLAYER)*['c']]
        MUT_PLICAT = []
        for b in BEST : MUT_PLICAT += (self.NB_CLUSTER-1)*[int(b)]
        SLC_2 = [list(BEST) + MUT_PLICAT + (self.NB_CLUSTER)*['new']]
        self.SLC = list(map(list, zip(*(SLC_1+SLC_2))))
        self.INFO_LOG.START_CYCLE(self.PLAYERS, self.ENV, self.GEN, self.SLC)
        ## Recurcivity OR ending
        self.SCORE_LIST = []
        if self.GEN != self.ARG[1] :
            return self.LAUNCH()
        else :
            TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            self.INFO_LOG.SAVE_CSV(TIME)
            print('TRAINNING FINISH')
        
    def SURVIVOR(self, BEST):
        old_PLAYERS = []
        for b in BEST :
            old_PLAYERS += [self.PLAYERS[b].RESET()]
        return old_PLAYERS
    
    def LEGACY(self, old_PLAYERS):
        mut_PLAYERS = []
        for p in old_PLAYERS :
            for i in range(self.NB_CLUSTER-1):
                mut_PLAYERS += [p.MUTATION()]
        return mut_PLAYERS
    
    def FOLLOWER(self):
        new_PLAYERS = []
        # density i/o considered
        DENSITY_IO = self.INFO_LOG.DENSITY_IO
        for i in range(self.NB_CLUSTER):
            new_PLAYERS += [Q_AGENT(*self.ARG[:-1], DENSITY_IO=DENSITY_IO)]
        return new_PLAYERS
    
    def ENV_UPDATE(self):
        self.ENV = []
        for p in self.PLAYERS :
            AGENT_VIEW, AGENT_MOVE = p.X, p.Y
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE))]
            p.INIT_ENV(self.ENV[-1])
        
if __name__ == '__main__' :
    EXP = LYFE(ARG_TUPLE)
    EXP.LAUNCH()
