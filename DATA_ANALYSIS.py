#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:47:40 2021
@author: fabien
"""

import numpy as np

import pandas as pd
import ast

from scipy.ndimage import filters

import EXTRA_FUNCTION as EF

################################ PARAMETER
# files
CSV_FILE = 'OUT/LYFE_RULE_0_EXP_20211125_145300_.csv'
CSV_FILE = 'OUT/LYFE_RULE_1_EXP_20211126_134230_.csv'
CSV_FILE = 'OUT/LYFE_RULE_2_EXP_20211125_132239_.csv'

CSV_FILE = 'OUT/LYFE_RULE_2_EXP_20211125_132239_.csv'

# animation
N_FRAME = None

################################ MODULE
class DATA_MEANING():
    def __init__(self, CSV_PATH):
        ### import csv 2 dataframe
        with open(CSV_PATH) as f:
            COL_NAME = f.readline().split('\n')[0].split(';')
            print(COL_NAME)
        col_type = {col : ast.literal_eval for col in COL_NAME}
        self.DF = pd.read_csv(CSV_PATH, sep=';', converters=col_type)
        # parameter extract
        self.ROWS, self.COLUMNS = self.DF.shape
        self.NB_GEN = self.DF['GEN'].max()+1
        self.NB_P_GEN = int(self.ROWS / self.NB_GEN)
        self.MAP_SIZE = int(self.DF['MAP_SIZE'].mean())
        # normalization
        s = self.DF['SCORE']
        self.DF['SCORE'] = (s-s.min())/(s.max()-s.min())
        ### groupby GEN
        self.GB_GEN = self.DF.groupby('GEN')
        # curve constructor
        self.TYPE_POS = DATA.GB_GEN.get_group(1).TYPE.values
        self.CURVE_CONSTRUCTOR = np.unique(self.TYPE_POS, return_counts=True, return_index=True)
        # label constructor (to upgrade)
        if True :
            self.TYPE = ['CTRL', 'PARENT', 'CHILD', 'RANDOM']
    
    def animate(self, NB_FRAME=True):
        # Generate data map per gen
        AGENT_POS, PNJ_POS = [], []
        for i, GG in self.GB_GEN :
            # AGENT POS (time, loc, xy)
            POS = np.array(list((map(list,GG['AGENT_POS']))))
            AGENT_POS += [np.moveaxis(POS, 0, 1)]
            # AGENT POS (time, loc, xy)
            POS = np.array(list((map(list,GG['PNJ_POS']))))
            PNJ_POS += [np.moveaxis(POS, 0, 1)]
        # Concatenate data
        AGENT_POS, PNJ_POS = np.concatenate(AGENT_POS), np.concatenate(PNJ_POS)
        # initialize anim
        self.ANIM_MAP = EF.MAP_ANIM(self.NB_P_GEN, self.MAP_SIZE)
        # add data
        self.ANIM_MAP.add_data([AGENT_POS, PNJ_POS])
        # animate
        #self.ANIM_MAP.animate(NB_FRAME)
        self.ANIM_MAP.video()

    def SCORE(self):
        ## plot part
        #self.GB_GEN.SCORE.describe()[['mean']].plot()
        curve_list, std_list = [], []
        # subplot list
        for v in self.CURVE_CONSTRUCTOR[0] :
            SubDF = self.DF[self.DF.TYPE == v].groupby('GEN')
            SCORE = SubDF.SCORE
            # curve and std
            curve_list += [filters.gaussian_filter1d(SCORE.mean().values,1)]
            std_list += [filters.gaussian_filter1d(SCORE.std().values,1)]
        EF.FAST_PLOT(curve_list,std_list,self.TYPE, 'TAG', 'SCORE','GEN', yaxis=[0.4,0.9])
        ## distrib part
        DIST_SCORE = []
        for i, s in self.GB_GEN.SCORE :
            DIST_SCORE += [s.values[None]]
        DIST_SCORE = np.concatenate(DIST_SCORE).T
        EF.FAST_IMSHOW([DIST_SCORE], ['DIST'])
        
    def EVOLUTION(self, SHOW_TREE=False):
        # convert tree lineage to imlineage
        self.PARENT = EF.TREELINEAGE_2_IMLINEAGE(self.GB_GEN)
        # parenting to networkx
        N_AGENT_TOT = self.NB_GEN*self.NB_P_GEN
        self.node = np.arange(N_AGENT_TOT)
        pos, self.G = EF.IMLINEAGE_2_GRAPH(self.node,self.PARENT)
        # calculate heritage
        self.G, edges_size, node_size, self.SHORT_PATH = EF.ADD_PATH(self.node,self.G)
        # prepare data indexes
        START,LENGHT = self.CURVE_CONSTRUCTOR[1:]
        ## show edges
        E_ = np.log(edges_size.reshape((self.NB_GEN,self.NB_P_GEN)).T+1)
        E_ = (E_-E_.min())/(E_.max()-E_.min())
        curve_e, std_e = EF.FAST_CURVE_CONSTRUCT(E_, [E_.mean(0), E_.std(0)], [START,LENGHT], 1)
        EF.FAST_PLOT(curve_e,std_e,self.TYPE+['MEAN'], 'TAG', 'EDGES','GEN', yaxis=[0.,1.])
        # show nodes
        N_ = np.sqrt(node_size.reshape((self.NB_GEN,self.NB_P_GEN)).T)
        YMAX = np.sqrt(N_.shape[1])
        curve_n, std_n = EF.FAST_CURVE_CONSTRUCT(N_, [N_.mean(0), N_.std(0)], [START,LENGHT], 1)
        EF.FAST_PLOT(curve_n,std_n,self.TYPE+['MEAN'], 'TAG', 'NODES','GEN', yaxis=[1.,YMAX])
        ## show image complete
        EF.FAST_IMSHOW([E_,N_], ['EDGES','NODE'])
        # plot tree
        if SHOW_TREE :
            EF.SHOW_TREE(self.G,pos)

if __name__ == '__main__' :
    # initialize
    DATA = DATA_MEANING(CSV_FILE)
    # score
    DATA.SCORE()
    # animate
    #DATA.animate()
    # network evolution
    DATA.EVOLUTION(False)