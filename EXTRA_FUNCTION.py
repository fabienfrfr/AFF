#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:28:43 2021
@author: fabien
"""

import numpy as np, pylab as plt
import matplotlib.animation as animation

from scipy.ndimage import filters

import os
import networkx as nx

from tqdm import tqdm

import skvideo.io as skio

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
        #self.DATA[2] += self.ORIGIN[None,:,:,None]
    
    def video(self):
        BIG_MAP_TIME = np.zeros((self.TIME_DURATION,self.X, self.Y))
        for i in range(self.TIME_DURATION):
            A = tuple(map(tuple,self.DATA[0][i].T))
            P = tuple(map(tuple,self.DATA[1][i].T))
            BIG_MAP_TIME[i][A] = 100
            BIG_MAP_TIME[i][P] = 200
        # export to video
        self.VIDEO = BIG_MAP_TIME
        FILENAME = os.getcwd() + os.path.sep + 'OUT' + os.path.sep + 'MAP_LYFE.mp4'
        skio.vwrite(FILENAME,self.VIDEO)
        
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
    X, Y = np.mgrid[0:NB_GEN, 0:NB_P_GEN]
    X, Y = X.reshape(-1), Y.reshape(-1)
    # generate Node
    for i in range(len(X)) :
        G.add_node(i,pos=(X[i],Y[i]))
    # genrate Edge
    for i in range(len(Y)) :
        GEN = abs(X[i])
        if GEN > 0 :
            y_loc = (Y == Y[i]-1)
            x_loc = (X == TREE_LIST[i][0])
            ID = np.where(y_loc*x_loc)[0]
            if len(ID) == 1 : G.add_edge(int(ID),i, weight = 0)
            else : G.add_edge(i,i, weight = 0)
        else :
            G.add_edge(i,i, weight = 0)
    # Extract pos dict
    pos = nx.get_node_attributes(G,'pos')
    return pos, G

def TREELINEAGE_2_IMLINEAGE(GROUBYGEN):
    PARENT = []
    for i,g in GROUBYGEN :
        # get tree info
        TREE = g.TREE
        T_ = []
        for t in TREE :
            if len(t) == 1 :
                T_ += t
            else :
                T_ += [t[0]]
        # normalize type
        T_ = np.array(T_)
        if i > 0 :
            NB_CONTROL = np.sum(g.TYPE.values == -1)
            T_ += NB_CONTROL
            T_[g.TYPE.values == -1] = np.arange(NB_CONTROL)
        else :
            T_[:] = -1
        T_[g.TYPE.values == 3] = -1
        PARENT += [T_[None]]
    #compile
    PARENT = np.concatenate(PARENT)
    return PARENT


def IMLINEAGE_2_GRAPH(node,PARENT):
    # extract info
    s = PARENT.shape
    pos = np.mgrid[:s[0],:s[1]].reshape((2,-1)).T
    NODE = node.reshape(s)
    # graph construction
    G = nx.DiGraph()
    for x,y,i in np.column_stack((pos,node[:,None])):
        G.add_node(i,pos=(x,y))
    for t in range(s[0]) :
        for i in range(s[1]):
            IDX = int(PARENT[t,i])
            if  IDX != -1 : 
                j, k = (IDX, t-1)
                G.add_edge(NODE[k,j] , NODE[t,i],  weight = 0)
                #G.add_edge(NODE[t,i], NODE[k,j] ,  weight = 0)
            else :
                G.add_edge(NODE[t,i], NODE[t,i],  weight = 0)
    # add number of child per node
    posD = nx.get_node_attributes(G,'pos')
    return posD, G

def SHOW_TREE(G,pos):
    H = G.to_undirected()
    nx.draw_networkx_nodes(H, pos, node_size=0.1, alpha=0.5)
    nx.draw_networkx_edges(H, pos, alpha=0.5)
    plt.savefig("OUT/TREE.svg")
    plt.show(); plt.close()

def ADD_PATH(node,G):
    SHORT_PATH = nx.shortest_path(G)
    node_size = np.array([len(values) for key,values in SHORT_PATH.items()])
    # Construct flow (for each edges beetween 2 nodes, add +1 if exist)
    closeness = np.array(list(nx.closeness_centrality(G).values()))
    closeness[closeness==closeness.min()] = closeness.max()
    edges_size = 1./(closeness+closeness.max())
    """
    for source in node :
        # listing of each link
        for target, target_list in SHORT_PATH[source].items() :
            # add +1 weight
            if target != source :
                s_,t_ = target_list[0:2]
                G[s_][t_]['weight'] += 1
    edges_size = np.array([G[u][v]['weight'] for u,v in G.edges])
    """
    return G, edges_size, node_size, SHORT_PATH

def FAST_CURVE_CONSTRUCT(TRAINING_, BEST_TRAINING, NB_SEED):
    curve_list = [  filters.gaussian_filter1d(TRAINING_[:1].mean(0),1), 
                    filters.gaussian_filter1d(TRAINING_[1:-NB_SEED].mean(0),1), 
                    filters.gaussian_filter1d(TRAINING_[-NB_SEED:].mean(0),1)]
    if True : curve_list += [filters.gaussian_filter1d(BEST_TRAINING,1)]
    std_list = [    filters.gaussian_filter1d(TRAINING_[:1].std(0),1),
                    filters.gaussian_filter1d(TRAINING_[1:-NB_SEED].std(0),1), 
                    filters.gaussian_filter1d(TRAINING_[-NB_SEED:].std(0),1)]
    if True : std_list += [np.zeros(BEST_TRAINING.size)]
    return curve_list,std_list

def FAST_PLOT(curve_list,std_list,label_list,Title,Ylabel,Xlabel, RULE=0, BATCH=0, CYCLE=0, NB=0, yaxis = None, XMAX=None):
    W, H, L, S = 3.7, 3.7, 18., 9. # width, height, label_size, scale_size
    #W, H, L, S = 3.7, 2.9, 18., 9. # width, height, label_size, scale_size
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
    if yaxis != None :
        plt.ylim(yaxis)
    # Save data
    plt.savefig('OUT' + os.path.sep + Title+Ylabel+Xlabel+"_r"+str(RULE)+"_b"
                +str(BATCH)+"n"+str(CYCLE)+"_nb"+str(NB)+ ".svg")
    plt.show(); plt.close()

def FAST_IMSHOW(img_list, name_list, stick=False):
    NB_PLOT = len(img_list)
    for i in range(NB_PLOT):
        fig = plt.figure(figsize=(3.7, 2.9))
        ax = fig.add_subplot()
        IM = img_list[i]
        im = plt.imshow(IM, interpolation='none', aspect="auto")
        # colorbar
        fig.colorbar(im, ax=ax, shrink=0.6)
        # Minor ticks
        if stick :
            Ly,Lx = IM.shape ; print(Lx,Ly)
            ax.set_xticks(np.arange(-0.5, Lx, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, Ly, 1), minor=True)
            ax.grid(which='minor', linewidth=10./Lx)
        # show
        plt.savefig('OUT' + os.path.sep + name_list[i]+'.svg')
        plt.show(); plt.close()

"""
def FAST_IMSHOW(img_list):
    NB_PLOT = len(img_list)
    fig, axarr = plt.subplots(1,NB_PLOT)
    if NB_PLOT == 1 : axarr = [axarr]
    for i in range(len(img_list)):
        cax = axarr[i].matshow(img_list[i], aspect='auto')
        fig.colorbar(cax, ax=axarr[i], shrink=0.6)
    plt.show(); plt.close()
"""

"""
node = np.arange(DATA.G.size())
H = DATA.G.copy()
SHORT_PATH = nx.shortest_path(H)
node_size = np.array([len(values) for key,values in SHORT_PATH.items()]) # closeness
# Construct flow (for each edges beetween 2 nodes, add +1 if exist)
for source in node :
    # listing of each link
    for target, target_list in SHORT_PATH[source].items() :
        # add +1 weight
        if target != source :
            s_,t_ = target_list[0:2]
            H[s_][t_]['weight'] += 1
                
edges_size = np.array([H[u][v]['weight'] for u,v in H.edges])

plt.imshow(np.log(edges_size.reshape((100,49)).T+1))


plt.imshow(node_size.reshape((100,49)).T)
source = 4789 # 22 daughter
"""