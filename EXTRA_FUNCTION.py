#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 18:07:48 2021
@author: fabien
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os 

FOLDER = 'OUT'
if(not os.path.isdir(FOLDER)): os.makedirs(FOLDER)

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

################################ ANIMATE  
class ANIM():
    def __init__(self, NB_GEN, MAP_SIZE):
        # init figure
        self.fig = plt.figure(figsize=(5,5), dpi=120) 
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.xticks([]), plt.yticks([])

        # Grid generator
        self.GRID, self.RESH = GRID_GEN(NB_GEN)

        # Image init
        self.MS, MAX_VAL = MAP_SIZE, 20
        self.X, self.Y = MAP_SIZE*np.array(self.GRID.shape)
        self.im = plt.imshow(MAX_VAL*np.random.random((self.X,self.Y)))
        # Data
        self.DATA = None
        self.LIST_DATA = []
        
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

    def gen_data(self,LENGHT, PLAYERS):
        self.DATA = np.zeros((LENGHT,self.X, self.Y), dtype=int)
        i = 0
        for p in PLAYERS:
            IMG_SEQ = p.MAP_LIST
            # find idx
            X,Y = np.where(self.GRID == i); X,Y = int(X), int(Y)
            self.DATA[:, X*self.MS:(X+1)*self.MS, Y*self.MS:(Y+1)*self.MS] = np.array(IMG_SEQ)
            i += 1
    
    def list_data(self):
        self.LIST_DATA += [self.DATA.copy()]
    
    def anim_update(self, i):
        self.im.set_array(self.DATA[i])
        return self.im

    def animate(self):
        self.DATA = np.concatenate(self.LIST_DATA, axis=0)
        self.anim = animation.FuncAnimation(self.fig, self.anim_update, frames=len(self.DATA), interval=1, blit=False)
        self.anim.save(filename=FOLDER + os.path.sep + 'OUTPUT_LYFE.mp4', writer='ffmpeg', fps=25) # png for alpha
        plt.show()
            
class PLOT_NET():
    def __init__(self, NB_GEN, IN):
        # Grid generator
        self.GRID, self.RESH = GRID_GEN(NB_GEN)
        # init figures (matplotlib.gridspec : if need flexible)
        self.fig, self.axes = plt.subplots(nrows=self.RESH[0], ncols=self.RESH[1])
        self.AX = None
        # neuron list
        self.GRAPH_LIST = []
        self.in_ = IN
        
    def gen_data(self, PLAYERS):
        self.GRAPH_LIST = []
        for p in PLAYERS :
            self.GRAPH_LIST += [p.AGENT.NEURON_LIST]
            
    def DRAW_NETWORK(self, NAME):
        ITER = 0
        for net_graph in self.GRAPH_LIST :
            # find idx :
            AX = np.where(self.GRID == ITER)
            self.AX = (AX[0][0], AX[1][0])
            ## Generate layer node
            # TRIPLET : (IDX, X, INDEX_ = Y)
            neuron_in, neuron_out = [], []
            # Input part :
            for n in range(self.in_):
                neuron_out += [[0,0,n]]
            # Layering
            for n in net_graph :
                # input part
                for n_ in range(n[2]) :
                    neuron_in  += [[n[0],n[3]-0.25,n_]]
                # output part
                for n_ in range(n[1]) :
                    neuron_out  += [[n[0],n[3]+0.25,n_]]
            neuron_in, neuron_out = np.array(neuron_in), np.array(neuron_out)
            ## Connect each Node
            for n in net_graph :
                i = 0
                for n_ in n[-1] :
                    connect_a = [n[3]-0.25, i]
                    idx = np.where((n_ == neuron_out[:, 0::2]).all(axis=1))[0]
                    connect_b = neuron_out[idx][:,1:]
                    X = np.concatenate((np.array([connect_a]), connect_b))
                    if X[0,0] > X[1,0] :
                        self.axes[self.AX].plot(X[:,0], X[:,1], 'k', lw=1, alpha=0.9)
                    else :
                        self.axes[self.AX].plot(X[:,0], X[:,1], 'r', lw=2, alpha=0.7)
                    # increment
                    i+=1
            ## Polygon neuron draw
            idx = np.unique(neuron_out[:,0])
            for i in idx :
                in_idx = np.where(i == neuron_in[:,0])[0]
                out_idx = np.where(i == neuron_out[:,0])[0]
                if in_idx.shape == (0,) :
                    x, y = np.max(neuron_out[out_idx,1:], axis=0)
                else :
                    x_i, y_i = np.max(neuron_in[in_idx,1:], axis=0)
                    x_o, y_o = np.max(neuron_out[out_idx,1:], axis=0)
                    x, y = np.mean((x_i, x_o)), np.max((y_i, y_o))
                # fill between polygon
                self.axes[self.AX].fill_between([x-0.5,x+0.5], [y+0.5,y+0.5], -0.5, alpha=0.5)
            ## Plot the graph-network
            self.axes[self.AX].scatter(neuron_in[:,1], neuron_in[:,2], s=10)
            self.axes[self.AX].scatter(neuron_out[:,1], neuron_out[:,2], s=30)
            ## Iteration
            ITER += 1
        self.fig.savefig(FOLDER + os.path.sep + 'NETWORK_'+str(NAME)+'.svg')

""""
plnet = PLOT_NET(9,9)
plnet.gen_data(EXP.PLAYERS)
plnet.DRAW_NETWORK()


"""
