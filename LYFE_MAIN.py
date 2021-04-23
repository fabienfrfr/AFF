#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:22:45 2021
@author: fabien
"""

import numpy as np

import time
import skvideo.io as io

from PLAY_IA import PLAY_IA
from ANIM_LYFE import ANIM, PLOT_NET

################################ EXPERIMENTAL PARAMETER (inverted name : need 2 change)
IO = (9,3)
NB_P_GEN = 2
batch_size = 16
MAP_SIZE = 12
N_TIME = 12
NB_GEN = 25

ARG_TUPLE = (IO,NB_P_GEN, batch_size, MAP_SIZE, N_TIME, NB_GEN)

################################ LYFE EXPERIMENT's 
class LYFE():
    def __init__(self, arg):
        self.NB_GEN = arg[-1]
        self.ARG = arg
        self.PLAYERS = []
        for n in range(self.NB_GEN) :
            self.PLAYERS += [PLAY_IA(*self.ARG[:-1])]
        # memory
        self.SCORE_LIST = []
        self.NETWORKING = [p.AGENT.NEURON_LIST.copy() for p in self.PLAYERS]
        self.GEN = 0
        self.TIME = [0,0]
        # for graphics
        self.PLNET = None
        self.ANIM = ANIM(self.NB_GEN, arg[3])
        # for next gen
        self.NB_CLUSTER = None
        self.NB_PLICAT = None

        
    def LAUNCH(self, VIDEO = False):
        self.TIME[0] = time.time()
        ## graph preview
        self.PLNET = PLOT_NET(self.NB_GEN,self.ARG[0][0])
        self.PLNET.gen_data(self.PLAYERS)
        self.PLNET.DRAW_NETWORK(self.GEN)
        ## party game
        for p in self.PLAYERS :
            p.PARTY()
            self.SCORE_LIST += [p.ENV.SCORE]
            print(self.SCORE_LIST[-1])
        ## construct image animation
        LENGHT = len(self.PLAYERS[0].MAP_LIST)
        self.ANIM.gen_data(LENGHT, self.PLAYERS)
        if VIDEO :
            io.vwrite("outputvideo.mp4", 255*self.ANIM.DATA)
        ## next_gen
        self.NB_CLUSTER = min(self.ANIM.RESH)
        self.NB_PLICAT = max(self.ANIM.RESH)
        # listing
        BEST = np.argsort(EXP.SCORE_LIST)[::-1]
        BEST = BEST[:self.NB_CLUSTER-1]
        old_PLAYERS = []
        # consevation
        for b in BEST :
            old_PLAYERS += [self.PLAYERS[b].RESET(*self.ARG[:-1])]
            # for time graph
            self.NETWORKING += [old_PLAYERS[-1].AGENT.NEURON_LIST.copy()]
        # mutation
        mut_PLAYERS = []
        for p in old_PLAYERS :
            for i in range(self.NB_PLICAT-1):
                mut_PLAYERS += [p.MUTATION(*self.ARG[:-1])]
                # for time graph
                self.NETWORKING += [mut_PLAYERS[-1].AGENT.NEURON_LIST.copy()]
        # new random
        new_PLAYERS = []
        for i in range(self.NB_PLICAT):
            new_PLAYERS += [PLAY_IA(*self.ARG[:-1])]
            # for time graph
            self.NETWORKING += [mut_PLAYERS[-1].AGENT.NEURON_LIST.copy()]
        # new players
        self.PLAYERS = old_PLAYERS + mut_PLAYERS + new_PLAYERS
        # reset param
        self.SCORE_LIST = []
        # time exp
        self.TIME[1] = time.time()
        print(self.TIME[1] - self.TIME[0])
        ## ending
        if self.GEN == self.ARG[1] :
            self.ANIM.animate()
        else :
            self.ANIM.list_data()
            self.GEN += 1
            return self.LAUNCH()
        
if __name__ == '__main__' :
    EXP = LYFE(ARG_TUPLE)
    EXP.LAUNCH()
