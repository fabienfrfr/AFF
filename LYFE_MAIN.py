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
IO = (17,3) # don't change here (not optimized yet)
NB_GEN = 50 # 100 convergence I/O when ?
batch_size = 64 #25 64
MAP_SIZE = 9
N_TIME = 16 #25
NB_P_GEN = 4**2 ## always squarable !

ARG_TUPLE = (IO,NB_GEN, batch_size, MAP_SIZE, N_TIME, NB_P_GEN)
RULE = 1 # 0 : classic

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
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE), self.GRULE, STOCK=True)]
            self.PLAYERS[-1].INIT_ENV(self.ENV[-1])
        # Classement & party info
        self.LOSS_LIST = []
        self.SCORE_LIST = []
        self.GEN = 0
        # Save-info
        self.INFO_LOG = LOG_INFO(self.PLAYERS, self.ENV, self.GEN, arg[1], rule, arg[2], arg[4])
        self.SLC, self.SLC_1, self.SLC_2 = None, None, None
        # for next gen (n-plicat) and control group
        self.NB_CONTROL = 1 # always (preference)
        self.NB_CHALLENGE = int(np.sqrt(self.NB_P_GEN)-self.NB_CONTROL)
        self.NB_SURVIVOR = self.NB_CHALLENGE # square completion
        self.NB_CHILD = int(np.sqrt(self.NB_P_GEN)-1)
        
    def LAUNCH(self, VIDEO = False):
        for o in tqdm(range(self.ARG[1]), position=0):
            P = (self.ARG[1] - o)/(2*self.ARG[1]) # proba
            ## party game
            for i in range(self.NB_P_GEN): #tqdm(range(self.NB_P_GEN), position=1, leave=None):
                self.PLAYERS[i].PARTY(self.ENV[i])
                self.LOSS_LIST += [np.mean(self.PLAYERS[i].LOSS[-5:])]
                self.SCORE_LIST += [self.ENV[i].SCORE]
            # fitness (loss and game score) stabilisation
            L, S = np.array(self.LOSS_LIST), np.array(self.SCORE_LIST)
            FITNESS =  (S-S.min())*(1-(L-L.min())/(L.max()-L.min()))
            # sort
            ORDER = np.argsort(FITNESS)[::-1]
            ORDER_ = np.argsort(FITNESS[self.NB_CONTROL:])[::-1]
            # complete cycle
            self.INFO_LOG.FINISH_CYCLE(self.ENV, self.SCORE_LIST, ORDER[::-1], self.PLAYERS)
            # density (after cycle) : forcing classing density in middle
            if np.random.choice((True,False), 1, [P,1-P]):
                self.INFO_LOG.DENSITY(self.PLAYERS, ORDER, (0,self.NB_P_GEN), IMSHOW=True)
            else :
                self.INFO_LOG.DENSITY(self.PLAYERS[self.NB_CONTROL:], ORDER_, (self.NB_CONTROL,self.NB_P_GEN), IMSHOW=True)
            # update gen
            self.GEN += 1
            #### CONTROL
            CTRL_PLAYER = self.PLAYERS[:self.NB_CONTROL]#; print(len(CTRL_PLAYER))
            #### NEXT GEN
            BEST = ORDER_[:self.NB_SURVIVOR]
            # SURVIVOR
            OLD_PLAYER = self.SURVIVOR(BEST, P)#; print(len(OLD_PLAYER))
            # MUTATION
            MUT_PLAYER = self.LEGACY(OLD_PLAYER)#; print(len(MUT_PLAYER))
            # CHALLENGER
            NEW_PLAYER = self.FOLLOWER()#; print(len(NEW_PLAYER))
            # UPDATE
            self.PLAYERS = CTRL_PLAYER + OLD_PLAYER + MUT_PLAYER + NEW_PLAYER
            for p in self.PLAYERS : p.LOSS = []
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
            self.LOSS_LIST = []
        #ennding
        self.COMPILE_SAVE()
        print('TRAINNING FINISH')

        
    def SURVIVOR(self, BEST, COMPLET_RESET):
        old_PLAYERS = []
        for b in BEST :
            old_PLAYERS += [self.PLAYERS[self.NB_CONTROL:][b].RESET(COMPLET_RESET)]
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
            if np.random.choice((True,False), 1, [0.65,0.35]) :
                new_PLAYERS += [Q_AGENT(*self.ARG[:-1], DENSITY_IO=DENSITY_IO)]
            else :
                new_PLAYERS += [Q_AGENT(*self.ARG[:-1])] # or not
        return new_PLAYERS
    
    def ENV_UPDATE(self):
        self.ENV = []
        for p in self.PLAYERS :
            AGENT_VIEW, AGENT_MOVE = p.X, p.Y
            self.ENV += [TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE), self.GRULE, STOCK=True)]
            p.INIT_ENV(self.ENV[-1])
    
    def COMPILE_SAVE(self):
        # save info
        TIME = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.INFO_LOG.SAVE_CSV(TIME)
        
if __name__ == '__main__' :
    # experiment
    EXP = LYFE(ARG_TUPLE, RULE)
    EXP.LAUNCH()

