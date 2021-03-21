#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:57:23 2021
@author: fabien
"""
import numpy as np
import pylab as plt

import skvideo.io as io

from Q_AGENT_GEN import Q_AGENT

TEST = True
import time

################################ TAG GAME ENVIRONMENT : BASIC's 
class TAG_ENV():
    def __init__(self, MAP_SIZE, AGENT_PROP):
        ## MAP INIT
        self.BACKGROUND = np.zeros((MAP_SIZE,MAP_SIZE))
        self.MAP_SIZE = MAP_SIZE
        ## PNJ and AGENT PLAYER POSITION
        self.PNJ_POS, self.AG_POS = self.POS_PLAYERS_FIRST()
        ## IT OR OT (for PNJ)
        self.IT = True
        ## MAP UPDATE
        self.MAP = None
        self.UPDATE_MAP()
        ## AGENT INFO
        self.AGENT_VIEW, self.AGENT_MOVE = AGENT_VIEW, AGENT_MOVE
        
    def POS_PLAYERS_FIRST(self):
        return np.random.randint(0,self.MAP_SIZE,2), np.random.randint(0,self.MAP_SIZE,2)
    
    def UPDATE_MAP(self):
        self.MAP = self.BACKGROUND.copy()
        if self.IT : A,B = 1., 2.
        else : A,B = 2., 1.
        self.MAP[tuple(self.AG_POS)] = A
        self.MAP[tuple(self.PNJ_POS)] = B
    
    def RESET(self) :
        ## Box observation
        BOX = np.mod(self.AGENT_VIEW + self.AG_POS, self.MAP_SIZE)
        prev_state = self.MAP[tuple(map(tuple, BOX.T))][np.newaxis]
        return prev_state
        
    def STEP(self, action) :
        ## UPDATE POS 
        # FOR "PNJ" :
        if self.IT :
            VECT = self.PNJ_POS - self.AG_POS
            COOR = np.where(abs(VECT)==abs(VECT).max())[0][0]
            if VECT[COOR] != 0 :
                self.PNJ_POS[COOR] -= np.sign(VECT[COOR])
        else :
            SIGN = np.random.randint(-1,2)
            COOR = np.random.randint(2)
            self.PNJ_POS[COOR] = np.mod(self.PNJ_POS[COOR] + SIGN, self.MAP_SIZE)
        # FOR "AGENT" :
        MVT = self.AGENT_MOVE[action]
        self.AG_POS = np.mod(self.AG_POS + MVT, self.MAP_SIZE)
        ## Update map
        self.UPDATE_MAP()
        ## Box observation
        BOX = np.mod(self.AGENT_VIEW + self.AG_POS, self.MAP_SIZE)
        new_state = self.MAP[tuple(map(tuple, BOX.T))][np.newaxis]
        ## Games rules (politics)
        GAP =  np.linalg.norm(self.PNJ_POS - self.AG_POS)
        reward = 0
        # PNJ is "IT"
        if self.IT :
            if GAP == 0 :
                reward = -10
                self.IT = np.invert(self.IT)
            elif GAP <= np.sqrt(2) :
                reward = -1
            elif GAP > 10 :
                reward = 1
        # AGENT is "IT"
        else :
            if GAP == 0 :
                reward = +10
                self.IT = np.invert(self.IT)
        ## ending condition
        DONE = False
        return new_state, reward, DONE
        
### TESTING PART
if TEST :
    t0 = time.time()
    # Parameter
    IO = (9,3)
    NB_P_GEN = 8
    batch_size = 16
    MAP_SIZE = 12
    N_TIME = 32
    ## Init
    AGENT = Q_AGENT(NB_P_GEN, IO, batch_size)
    AGENT_VIEW, AGENT_MOVE = AGENT.X, AGENT.Y
    
    ENV = TAG_ENV(MAP_SIZE, (AGENT_VIEW, AGENT_MOVE))
    # first step
    prev_state = ENV.RESET()
    # RL Loop
    MAP_ = [ENV.MAP]
    for t in range(N_TIME) :
        for i in range(batch_size):
            action = AGENT.ACTION(prev_state)
            new_state, reward, DONE = ENV.STEP(action)
            # Memory update
            if i == batch_size-1 : DONE = True
            AGENT.SEQUENCING(prev_state,action,new_state,reward,DONE)
            MAP_ += [ENV.MAP]
            # n+1
            prev_state = new_state.copy()
        # Reinforcement learning
        AGENT.OPTIM()
    # mvt video
    io.vwrite("outputvideo.mp4", 255*np.array(MAP_))
    # final
    plt.imshow(MAP_[0]); plt.show(); plt.close()
    plt.imshow(MAP_[-1]); plt.show(); plt.close()
    t1 = time.time()
    print(t1-t0)
