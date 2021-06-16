#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:57:23 2021
@author: fabien
"""
import numpy as np

################################ TAG GAME ENVIRONMENT : BASIC's 
class TAG_ENV():
    def __init__(self, MAP_SIZE, AGENT_PROP):
        ## subMAP INIT
        self.BACKGROUND = np.zeros((MAP_SIZE,MAP_SIZE))
        self.MAP_SIZE = MAP_SIZE
        ## PNJ and AGENT PLAYER POSITION
        self.PNJ_POS, self.AG_POS = self.POS_PLAYERS_FIRST()
        self.PNJ_LIST, self.AG_LIST = [list(self.PNJ_POS)], [list(self.AG_POS)]
        ## IT OR OT (for PNJ)
        self.IT = True
        self.IT_LIST = [self.IT]
        ## MAP UPDATE
        self.MAP = None
        self.MAP_LIST = []
        self.UPDATE_MAP()
        ## AGENT INFO
        self.AGENT_VIEW, self.AGENT_MOVE = AGENT_PROP
        ## ENDING
        self.SCORE = 0
        self.END = np.inf
        
    def POS_PLAYERS_FIRST(self):
        return np.random.randint(0,self.MAP_SIZE,2), np.random.randint(0,self.MAP_SIZE,2)
    
    def UPDATE_MAP(self):
        self.MAP = self.BACKGROUND.copy()
        if self.IT : A,B = 10., 20.
        else : A,B = 20., 10.
        self.MAP[tuple(self.AG_POS)] = A
        self.MAP[tuple(self.PNJ_POS)] = B
        self.MAP_LIST += [self.MAP.copy()]
    
    def RESET(self) :
        ## Box observation
        BOX = np.mod(self.AGENT_VIEW + self.AG_POS, self.MAP_SIZE)
        prev_state = self.MAP[tuple(map(tuple, BOX.T))][np.newaxis]
        return prev_state
        
    def STEP(self, action) :
        ## UPDATE POS 
        # FOR "PNJ" :
        VECT = self.PNJ_POS - self.AG_POS
        if self.IT :
            COOR = np.where(abs(VECT)==abs(VECT).max())[0][0]
            if VECT[COOR] != 0 :
                self.PNJ_POS[COOR] -= np.sign(VECT[COOR])
        else :
            if np.random.choice([True, False], p=[0.9,0.1]) :
                COOR = np.where(abs(VECT)==abs(VECT).min())[0][0]
                SIGN = np.sign(VECT[COOR])
            else :
                COOR = np.random.randint(2)
                SIGN = np.random.randint(-1,2)
            self.PNJ_POS[COOR] = np.mod(self.PNJ_POS[COOR] - SIGN, self.MAP_SIZE)
        self.PNJ_LIST += [list(self.PNJ_POS)]
        # FOR "AGENT" :
        MVT = self.AGENT_MOVE[action]
        self.AG_POS = np.mod(self.AG_POS + MVT, self.MAP_SIZE)
        self.AG_LIST += [list(self.AG_POS)]
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
            elif GAP <= np.sqrt(2) :
                reward = +1
            elif GAP > 10 :
                reward = -1
        self.IT_LIST += [self.IT]
        ## ending condition
        self.SCORE += reward
        if abs(self.SCORE) > self.END :
            DONE = True
        else :
            DONE = False
        return new_state, reward, DONE