#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:28:43 2021
@author: fabien
"""

import numpy as np, pylab as plt
import matplotlib.animation as animation

import os
import networkx as nx

from tqdm import tqdm

################################ EXTRA FUNCTION 
def PrimeFactors(n):
    LIST = []
    # pairs
    while n % 2 == 0:
        LIST += [2]
        n = n / 2
    # odd
    for i in range(3,int(np.sqrt(n))+1,2):
        while n % i == 0:
            LIST += [i]
            n = n / i
    # is prime
    if n > 2: LIST += [n]
    return np.array(LIST, dtype=int)

def GRID_GEN(NB_GEN) :
    # Grid generator
    GRID = np.arange(NB_GEN)
    RESH = PrimeFactors(NB_GEN)
    if RESH.size == 1 :
        RESH = [1] + list(RESH)
        GRID = GRID[None]
    elif RESH.size == 2 :
        GRID = GRID.reshape(RESH)
    elif RESH.size == 3 :
        RESH = (np.product(RESH[:2]),RESH[-1])
        GRID = GRID.reshape(RESH)
    else :
        CUT = np.ceil(RESH.size/2).astype(int)
        RESH = (np.product(RESH[:CUT]),np.product(RESH[CUT:]))
        GRID = GRID.reshape(RESH)
    RESH = tuple(RESH)
    return (GRID,RESH)

class MAP_ANIM():
    def __init__(self, NB_P_GEN, MAP_SIZE, STEP_LINE = 3):
        # init figure
        self.fig = plt.figure(figsize=(1,1), dpi=60) 
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.xticks([]), plt.yticks([])
        # Grid generator
        self.GRID, self.RESH = GRID_GEN(NB_P_GEN)
        # Image init
        self.MAP_SIZE = MAP_SIZE
        self.X, self.Y = MAP_SIZE*np.array(self.RESH)
        self.BIG_MAP = np.zeros((self.X, self.Y))
        self.im = plt.imshow(self.BIG_MAP, cmap='Greys', vmax = np.pi)
        # agent & pnj position
        A,B = MAP_SIZE*np.mgrid[:self.RESH[0], :self.RESH[1]].reshape(2,-1)
        self.point_pnj = plt.scatter(A,B, c='C0')
        self.point_agt = plt.scatter(A,B, c='C1')
        # agent & pnj line
        self.STEP = STEP_LINE
        self.lines_pnj = self.line_gen(NB_P_GEN, 'C0')
        self.lines_agt = self.line_gen(NB_P_GEN, 'C1')        
        # Data
        self.DATA = None
        self.TIME_DURATION = None
        self.pbar = tqdm(total=100)
        self.ORIGIN = None
        # Major ticks
        self.ax.set_xticks(np.arange(0., self.X, MAP_SIZE))
        self.ax.set_yticks(np.arange(0., self.Y, MAP_SIZE))
        # parametrisation plot
        plt.xlim(0.5,self.X-0.5); plt.ylim(-0.5,self.Y-0.5)
        # Delete border
        self.ax.spines['bottom'].set_color('None'); self.ax.spines['top'].set_color('None') 
        self.ax.spines['right'].set_color('None'); self.ax.spines['left'].set_color('None')
        # show grid
        self.ax.grid(True)
    
    def line_gen(self,NB_P_GEN, COLOR):
        lines = []
        for index in range(NB_P_GEN):
            ax_ = self.ax.plot([],[],lw=2,color=COLOR)[0]
            lines.append(ax_)
        return lines

    def add_data(self,DATA_LIST):
        self.DATA = DATA_LIST
        self.TIME_DURATION = len(DATA_LIST[0])
        # real position in map
        self.ORIGIN = self.MAP_SIZE*np.mgrid[:self.RESH[0], :self.RESH[1]].reshape(2,-1).T
        self.DATA[0] += self.ORIGIN[None,:,:]
        self.DATA[1] += self.ORIGIN[None,:,:]
        self.DATA[2] += self.ORIGIN[None,:,:,None]
    
    def anim_update(self, i):
        # extract data
        A, P, I = self.DATA
        # update img
        self.BIG_MAP = np.zeros((self.X, self.Y))
        VIEW_IDX = np.moveaxis(I[i], 0,1).reshape(2,-1)
        self.BIG_MAP[tuple(map(tuple,VIEW_IDX))] = 1
        self.im.set_array(self.BIG_MAP.T)
        # update line
        if i < self.STEP : i_ = 0
        else : i_ = i - self.STEP
        for lnum,line_p in enumerate(self.lines_pnj):
            line_p.set_data(P[i_:i,lnum,0], P[i_:i,lnum,1])
        for lnum,line_a in enumerate(self.lines_agt):
            line_a.set_data(A[i_:i,lnum,0], A[i_:i,lnum,1])
        # update point
        self.point_pnj.set_offsets(P[i])
        self.point_agt.set_offsets(A[i])
        if i%int(self.TIME_DURATION/100) == 0 :
            self.pbar.update(1)
        elif i == self.TIME_DURATION :
            self.pbar.update(1)
            self.pbar.close()
        return self.im, self.point_pnj, self.point_agt, self.lines_pnj, self.lines_agt

    def animate(self, FRAME = True):
        if (FRAME == True) or (FRAME > self.TIME_DURATION) :
            FRAME = self.TIME_DURATION
        self.anim = animation.FuncAnimation(self.fig, self.anim_update, frames=FRAME, interval=1, blit=False)
        self.anim.save(filename= os.getcwd() + os.path.sep + 'OUT' + os.path.sep + 'MAP_LYFE.mp4', writer='ffmpeg', fps=25) # png for alpha
        plt.savefig(os.getcwd() + os.path.sep + 'OUT' + os.path.sep + 'MAP_LYFE.png', dpi = 360)
        plt.show()

def NEURON_2_GRAPH(NEURON_INFO, IO, LINK_LAYERS = True):
    # Create new graph
    G = nx.DiGraph()
    # 4-PLET : (IDX, INDEX_LAYERS, INDEX_NODE = Y, X)
    NEURON_IN, NEURON_OUT = [], []
    # Input part :
    for n in range(IO[0]):
        NEURON_OUT += [[0,n,0]]
    # Layering
    for n in NEURON_INFO :
        # input part
        for n_ in range(n[2]) :
            NEURON_IN  += [[n[0],n_,n[3]-0.25]]
        # output part
        for n_ in range(n[1]) :
            NEURON_OUT  += [[n[0],n_,n[3]+0.25]]
    NEURON_IN, NEURON_OUT = np.array(NEURON_IN), np.array(NEURON_OUT)
    # Node construction
    NEURON_IO = np.concatenate((NEURON_IN, NEURON_OUT))
    for i in range(len(NEURON_IO)) :
        y,x = NEURON_IO[i][1:]
        G.add_node(i, pos=(x,y))
    ## Connect each Node
    for n in NEURON_INFO :
        for i in range(n[2]) :
            # Input
            n_i = [n[0],i]
            idx_i = np.where((n_i == NEURON_IN[:, :-1]).all(axis=1))[0]
            # Output
            n_o = n[-1][i]
            idx_o = np.where((n_o == NEURON_OUT[:, :-1]).all(axis=1))[0]
            ## Edge
            G.add_edge(int(idx_o)+len(NEURON_IN),int(idx_i))
            # Link Layers
            if LINK_LAYERS :
                for j in np.where(NEURON_OUT[:,0] == n[0])[0]:
                    G.add_edge(int(idx_i),int(j)+len(NEURON_IN))        
    # Extract pos dict
    pos = nx.get_node_attributes(G,'pos')
    return pos, G

def LINEAGE_2_GRAPH(NB_GEN,NB_P_GEN,TREE_LIST):
    # Create new graph
    G = nx.DiGraph()
    # Positionnal
    Y, X = np.mgrid[0:NB_GEN, 0:NB_P_GEN]
    Y, X = Y.reshape(-1), X.reshape(-1)
    # generate Node
    for i in range(len(X)) :
        G.add_node(i,pos=(X[i],-Y[i]))
    # genrate Edge
    for i in range(len(X)) :
        GEN = abs(Y[i])
        if GEN > 0 :
            y_loc = (Y == Y[i]-1)
            x_loc = (X == TREE_LIST[i][0])
            ID = np.where(y_loc*x_loc)[0]
            if len(ID) == 1 : G.add_edge(int(ID),i, weight = 0)
    # Extract pos dict
    pos = nx.get_node_attributes(G,'pos')
    return pos, G

def FAST_PLOT(curve_list,std_list,label_list,Title,Ylabel,Xlabel, RULE=0, BATCH=0, CYCLE=0, NB=0, XMAX=None):
    print(XMAX)
    W, H, L, S = 3.7, 2.9, 18., 9. # width, height, label_size, scale_size
    # fig ratio
    MM2INCH = 1# 2.54
    W, H, L, S = np.array((W, H, L, S))/MM2INCH # ratio fig : 2.7/2.1
    STD = np.pi
    # Figure
    fig = plt.figure(figsize=(W, H))
    
    plt.rc('font', size=S)
    plt.rc('axes', titlesize=S)
    
    ax = fig.add_subplot()
    ax.set_title(Title, fontsize=L)
    ax.set_ylabel(Ylabel, fontsize=L)
    ax.set_xlabel(Xlabel, fontsize=L)
    # ax loop
    for c,s,l in zip(curve_list,std_list, label_list) :
        ax.plot(c, label=l)
        ax.fill_between(np.arange(len(c)), c - s/STD, c + s/STD, alpha=0.3)
    # Legend
    ax.legend()
    if XMAX == None : 
        plt.xlim([0,len(c)])
    else :
        plt.xlim([0,XMAX])
    # Save data
    plt.savefig('OUT' + os.path.sep + Title+Ylabel+Xlabel+"_r"+str(RULE)+"_b"
                +str(BATCH)+"n"+str(CYCLE)+"_nb"+str(NB)+ ".svg")
    plt.show(); plt.close()

def FAST_IMSHOW(img_list):
    fig, axarr = plt.subplots(1,len(img_list))
    for i in range(len(img_list)):
        cax = axarr[i].imshow(img_list[i])
        fig.colorbar(cax, ax=axarr[i], shrink=0.6)
    plt.show(); plt.close()