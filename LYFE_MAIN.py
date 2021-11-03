#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:22:45 2021
@author: fabien
"""

import numpy as np

import datetime
#import skvideo.io as io

from Q_AGENT import Q_AGENT
from TAG_ENV import TAG_ENV
from LOG_GEN import LOG_INFO

from tqdm import tqdm

################################ EXPERIMENTAL PARAMETER (inverted name : need 2 change)
IO = (13,3) # don't change here (not optimized yet)
NB_GEN = 100 # convergence I/O vers 20
batch_size = 25 #25
MAP_SIZE = 9
N_TIME = 25 #25
NB_P_GEN = 5**2 ## always squarable !

ARG_TUPLE = (IO,NB_GEN, batch_size, MAP_SIZE, N_TIME, NB_P_GEN)
RULE = 0 # 0 : classic

################################ LYFE EXPERIMENT's 
class LYFE():
    def __init__(self, arg, rule=0):
        # Parameter
        self.NB_P_GEN = arg[-1]
        self.MAP_SIZE = arg[3]
        self.ARG = arg
        self.GRULE = rule
        # Loop generate player and env (change if common of separated)
        self.PLAYERS, self.ENV = [], []
        for n in range(self.NB_P_GEN) :
            if n == 0 :
                self.PLAYERS += [Q_AGENT(*self.ARG[:-1], CTRL=True)]
            else :
                self.PLAYERS += [Q_AGENT(*self.ARG[:-1])]
            AGENT_VIEW, AGENT_MOVE = self.PLAYERS[-1].X, self.PLAYERS[-1].Y
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE), self.GRULE)]
            self.PLAYERS[-1].INIT_ENV(self.ENV[-1])
        # Classement & party info
        self.SCORE_LIST = []
        self.GEN = 0
        # Save-info
        self.INFO_LOG = LOG_INFO(self.PLAYERS, self.ENV, self.GEN)
        self.SLC, self.SLC_1, self.SLC_2 = None, None, None
        # for next gen (n-plicat) and control group
        self.NB_CONTROL = 1 # always (preference)
        self.NB_CHALLENGE = int(np.sqrt(self.NB_P_GEN)-self.NB_CONTROL)
        self.NB_SURVIVOR = self.NB_CHALLENGE # square completion
        self.NB_CHILD = int(np.sqrt(self.NB_P_GEN)-1)
        
    def LAUNCH(self, VIDEO = False):
        for _o in tqdm(range(self.ARG[1]), position=0):
            ## party game
            for i in tqdm(range(self.NB_P_GEN), position=1, leave=None):
                self.PLAYERS[i].PARTY(self.ENV[i])
                self.SCORE_LIST += [self.ENV[i].SCORE]
            ## Pre-analysis
            ORDER = np.argsort(self.SCORE_LIST)[::-1]
            ORDER_ = np.argsort(self.SCORE_LIST[self.NB_CONTROL:])[::-1]
            # complete cycle
            self.INFO_LOG.FINISH_CYCLE(self.ENV, self.SCORE_LIST, ORDER[::-1])
            # density (after cycle)
            self.INFO_LOG.DENSITY(self.PLAYERS[self.NB_CONTROL:], ORDER_, (self.NB_CONTROL,self.NB_P_GEN), IMSHOW=True)
            # update gen
            self.GEN += 1
            #### CONTROL
            CTRL_PLAYER = self.PLAYERS[:self.NB_CONTROL]#; print(len(CTRL_PLAYER))
            #### NEXT GEN
            BEST = ORDER_[:self.NB_SURVIVOR]
            # SURVIVOR
            OLD_PLAYER = self.SURVIVOR(BEST)#; print(len(OLD_PLAYER))
            # MUTATION
            MUT_PLAYER = self.LEGACY(OLD_PLAYER)#; print(len(MUT_PLAYER))
            # CHALLENGER
            NEW_PLAYER = self.FOLLOWER()#; print(len(NEW_PLAYER))
            # UPDATE
            self.PLAYERS = CTRL_PLAYER + OLD_PLAYER + MUT_PLAYER + NEW_PLAYER
            self.ENV_UPDATE()
            ## Re-init cycle
            SLC_1 = [self.NB_CONTROL*['c']+len(OLD_PLAYER)*['s']+len(MUT_PLAYER)*['l']+len(NEW_PLAYER)*['f']]#; print(SLC_1)
            MUT_PLICAT = []
            for b in BEST : MUT_PLICAT += (self.NB_CHILD)*[int(b)]
            SLC_2 = [(self.NB_CONTROL)*['ctrl'] + list(BEST) + MUT_PLICAT + (self.NB_CHALLENGE)*['new']]#; print(SLC_2)
            self.SLC = list(map(list, zip(*(SLC_1+SLC_2))))
            self.INFO_LOG.START_CYCLE(self.PLAYERS, self.ENV, self.GEN, self.SLC)
            ## Recurcivity OR ending
            self.SCORE_LIST = []
        TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.INFO_LOG.SAVE_CSV(TIME)
        print('TRAINNING FINISH')
        
    def SURVIVOR(self, BEST):
        old_PLAYERS = []
        for b in BEST :
            old_PLAYERS += [self.PLAYERS[self.NB_CONTROL:][b].RESET()]
        return old_PLAYERS
    
    def LEGACY(self, old_PLAYERS):
        mut_PLAYERS = []
        # density i/o considered
        DENSITY_IO = self.INFO_LOG.DENSITY_IO
        for p in old_PLAYERS :
            for i in range(self.NB_CHILD):
                mut_PLAYERS += [p.MUTATION(DENSITY_IO)]
        return mut_PLAYERS
    
    def FOLLOWER(self):
        new_PLAYERS = []
        # density i/o considered
        DENSITY_IO = self.INFO_LOG.DENSITY_IO
        for i in range(self.NB_CHALLENGE):
            # follower :
            if np.random.randint(2, dtype=bool) :
                new_PLAYERS += [Q_AGENT(*self.ARG[:-1], DENSITY_IO=DENSITY_IO)]
            else :
                new_PLAYERS += [Q_AGENT(*self.ARG[:-1])] # or not
        return new_PLAYERS
    
    def ENV_UPDATE(self):
        self.ENV = []
        for p in self.PLAYERS :
            AGENT_VIEW, AGENT_MOVE = p.X, p.Y
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE), self.GRULE)]
            p.INIT_ENV(self.ENV[-1])
        
if __name__ == '__main__' :
    # experiment
    EXP = LYFE(ARG_TUPLE, RULE)
    EXP.LAUNCH()
