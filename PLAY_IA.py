#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:22:45 2021
@author: fabien
"""

from Q_AGENT_GEN import Q_AGENT
from TAG_ENV import TAG_ENV

################################ TAG GAME ENVIRONMENT : BASIC's 
class PLAY_IA():
    def __init__(self, *arg, AGENT = None):
        # GAME Parameter
        self.IO = arg[0]
        self.NB_P_GEN = arg[1]
        self.batch_size = arg[2]
        self.MAP_SIZE = arg[3]
        self.N_TIME = arg[4]
        ## Init AGENT & ENV
        if AGENT == None :
            self.AGENT = Q_AGENT(self.NB_P_GEN, self.IO, self.batch_size)
        else : 
            self.AGENT = AGENT
        AGENT_VIEW, AGENT_MOVE = self.AGENT.X, self.AGENT.Y
        self.ENV = TAG_ENV(self.MAP_SIZE, (AGENT_VIEW, AGENT_MOVE))
        # First step
        self.prev_state = self.ENV.RESET()
        self.MAP_LIST = [self.ENV.MAP]

    def PARTY(self):
        for t in range(self.N_TIME):
            for i in range(self.batch_size):
                action = self.AGENT.ACTION(self.prev_state)
                new_state, reward, DONE = self.ENV.STEP(action)
                # Memory update
                if i == self.batch_size-1 : DONE = True
                self.AGENT.SEQUENCING(self.prev_state,action,new_state,reward,DONE)
                #AGENT.REWARD(reward)
                self.MAP_LIST += [self.ENV.MAP]
                # n+1
                self.prev_state = new_state.copy()
                # escape loop
                if DONE == True : break                
            # Reinforcement learning
            self.AGENT.OPTIM()
    
    def RESET(self, *arg):
        AGENT = self.AGENT.RESET()
        return PLAY_IA(*arg, AGENT = AGENT)
    
    def MUTATION(self, *arg):
        AGENT = self.AGENT.LAUNCH_MUTATION()
        return PLAY_IA(*arg, AGENT = AGENT)
