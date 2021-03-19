#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:57:23 2021
@author: fabien
"""
import numpy as np
import pylab as plt

from Q_AGENT_GEN import Q_AGENT

TEST = True

"""
version 1 : Only one "IT" players, the agents don't become "IT"
version 2 : The agent can becaume "IT"
"""

################################ TAG GAME ENVIRONMENT : BASIC's 
class TAG_ENV():
    def __init__(self, MAP_SIZE, AGENT_VIEW, AGENT_MOVE):
        ## MAP INIT
        self.BACKGROUND = np.zeros((MAP_SIZE,MAP_SIZE))
        self.MAP_SIZE = MAP_SIZE
        ## IT and AGENT PLAYER POSITION
        self.IT_POS, self.AG_POS = self.POS_PLAYERS_FIRST()
        ## MAP UPDATE
        self.MAP = None
        self.UPDATE_MAP()
        ## AGENT INFO
        self.AGENT_VIEW, self.AGENT_MOVE = AGENT_VIEW, AGENT_MOVE
        
    def POS_PLAYERS_FIRST(self):
        return np.random.randint(0,self.MAP_SIZE,2), np.random.randint(0,self.MAP_SIZE,2)
    
    def UPDATE_MAP(self):
        self.MAP = self.BACKGROUND.copy()
        self.MAP[tuple(self.AG_POS)] = 1.
        self.MAP[tuple(self.IT_POS)] = 2.
    
    def RESET(self) :
        return None
        
    def STEP(self, action) :
        ## UPDATE POS 
        # FOR "IT" :
        VECT = self.IT_POS - self.AG_POS
        COOR = np.where(abs(VECT)==abs(VECT).max())[0][0]
        if VECT[COOR] != 0 :
            self.IT_POS[COOR] -= np.sign(VECT[COOR])
        # FOR "AGENT" :
        MVT = self.AGENT_MOVE[action]
        self.AG_POS = np.mod(self.AG_POS + MVT, self.MAP_SIZE)
        ## Update map
        self.UPDATE_MAP()
        ## Box observation
        BOX = np.mod(self.AGENT_VIEW + self.AG_POS, MAP_SIZE)
        new_state = self.MAP[tuple(map(tuple, BOX.T))]
        print(new_state)
        ## Politics
        GAP =  np.linalg.norm(self.IT_POS - self.AG_POS)
        if GAP == 0 :
            reward = -10
        elif GAP <= np.sqrt(2) :
            reward = -1
        elif GAP > 10 :
            reward = 1
        else :
            reward = 0
        ## ending condition
        DONE = False
        return new_state, reward, DONE
        
### TESTING PART
if TEST :
    # Parameter
    IO = (9,3)
    NB_P_GEN = 8
    batch_size = 16
    MAP_SIZE = 12
    ## Init
    AGENT = Q_AGENT(NB_P_GEN, IO, batch_size)
    AGENT_VIEW, AGENT_MOVE = AGENT.X, AGENT.Y
    
    ENV = TAG_ENV(MAP_SIZE, AGENT_VIEW, AGENT_MOVE)
    plt.imshow(ENV.MAP); plt.show();plt.close()
    new_state, reward, DONE = ENV.STEP(1)
    plt.imshow(ENV.MAP)
