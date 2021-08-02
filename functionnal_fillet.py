#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:09:28 2021
@author: fabien
"""

class model():
    def __init__(self, BATCH_SIZE, NB_GEN, NB_SEEDER):
        # Parameter
        self.BATCH_SIZE = BATCH_SIZE
        self.NB_GEN = NB_GEN
        self.NB_SEEDER = NB_SEEDER
        # generate first ENN step
        self.SEEDER_LIST = []
        
    def fit(self, DATA, LABEL):
        # Store data/label
        self.DATA = DATA
        self.LABEL = LABEL
        # Training Loop :
        for i in range(self.NB_GEN):
            for j in range(self.NB_SEEDER):
                # update seeder list
                print(i,j)
    
    def predict(self, DATA_NEW):
        # Put data in best model
        self.LABEL_NEW = self.SEEDER_LIST[-1](DATA_NEW)
        # return result
        return self.LABEL_NEW