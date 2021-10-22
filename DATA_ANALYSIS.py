#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 09:47:40 2021
@author: fabien
"""

import numpy as np, pylab as plt
import matplotlib.gridspec as gridspec

import pandas as pd, scipy as sp
import ast
import networkx as nx

import EXTRA_FUNCTION as EF

################################ PARAMETER
# files
CSV_FILE = 'OUT/LYFE_EXP_20211022_184030_.csv' # 'OUT/LYFE_EXP_20210509_163710_.csv'
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
        ### groupby GEN
        self.GB_GEN = self.DF.groupby('GEN')
    
    def animate(self, NB_FRAME=True):
        # Generate data map per gen
        AGENT_POS, PNJ_POS, IN_VIEW = [], [], []
        for i, GG in self.GB_GEN :
            # AGENT POS (time, loc, xy)
            POS = np.array(list((map(list,GG['AGENT_POS']))))
            AGENT_POS += [np.moveaxis(POS, 0, 1)]
            # AGENT POS (time, loc, xy)
            POS = np.array(list((map(list,GG['PNJ_POS']))))
            PNJ_POS += [np.moveaxis(POS, 0, 1)]
            # NB ITER 
            PERIOD = POS.shape[1]
            # AGENT VIEW (time, loc, xy, id)
            POS = np.array([list((map(list,GG['IN_COOR'].values)))]*PERIOD)
            IN_VIEW += [np.moveaxis(POS, -2, -1)]
        # Concatenate data
        AGENT_POS, PNJ_POS, IN_VIEW = np.concatenate(AGENT_POS), np.concatenate(PNJ_POS), np.concatenate(IN_VIEW)
        # View per Grid
        IN_VIEW += AGENT_POS[:,:,:,None]
        IN_VIEW = IN_VIEW % self.MAP_SIZE
        # initialize anim
        ANIM_MAP = EF.MAP_ANIM(self.NB_P_GEN, self.MAP_SIZE)
        # add data
        ANIM_MAP.add_data([AGENT_POS, PNJ_POS, IN_VIEW])
        # animate
        ANIM_MAP.animate(NB_FRAME)
    
    def SCORE(self):
        self.GB_GEN.SCORE.describe()[['mean']].plot()
        # only control
        X = self.DF[self.DF.TYPE == -1].GEN
        Y_CONTROL = self.DF[self.DF.TYPE == -1].SCORE
        plt.plot(X,Y_CONTROL); plt.show();plt.close()
    
    def DENSITY_T(self):
        D_IN, D_OUT = [], []
        for i, GG in self.GB_GEN :
            # extrat in data
            DENSITY_IN = GG['DENSITY_IN']
            DENSITY_OUT = GG['DENSITY_OUT']
            # increment
            D_IN += [DENSITY_IN.values[0]]
            D_OUT += [DENSITY_OUT.values[0]]
        # List 2 Numpy array
        D_IN, D_OUT = np.array(D_IN), np.array(D_OUT)
        ### plot
        fig = plt.figure(constrained_layout=True)
        # grid
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax00 = fig.add_subplot(gs[0, 0])
        ax01 = fig.add_subplot(gs[0, 1])
        ax10 = fig.add_subplot(gs[1, 0])
        ax11 = fig.add_subplot(gs[1, 1])
        # show
        ax00.imshow(D_IN.T,cmap='Greys', origin='lower', aspect="auto")
        ax01.imshow(D_IN[-1].reshape(5,5),cmap='Greys', origin='lower', aspect="auto")
        ax10.imshow(D_OUT.T,cmap='Greys', origin='lower', aspect="auto")
        ax11.imshow(D_OUT[-1].reshape(3,3),cmap='Greys', origin='lower', aspect="auto")
        plt.show();plt.close()

if __name__ == '__main__' :
    # initialize
    DATA = DATA_MEANING(CSV_FILE)
    # score
    DATA.SCORE()
    # strategy i/o
    DATA.DENSITY_T()
    # animate
    DATA.animate()
    



"""
################################ ALGO

### General parameter EXTRACT
ROWS, COLUMNS = DF.shape
NB_GEN = DF['GEN'].max()+1
NB_P_GEN = int(ROWS / NB_GEN)
MAP_SIZE = int(DF['MAP_SIZE'].mean())

NB_BEST = int(np.sqrt(NB_P_GEN)-1)
GB_GEN = DF.groupby('GEN')

### DENSITY CARACTERISATION
# List of density
D_IN, D_OUT = [], []
for i, GG in GB_GEN :
    # extrat in data
    DENSITY_IN = GG['DENSITY_IN']
    DENSITY_OUT = GG['DENSITY_OUT']
    # increment
    D_IN += [DENSITY_IN.values[0]]
    D_OUT += [DENSITY_OUT.values[0]]
# List 2 Numpy array
D_IN, D_OUT = np.array(D_IN), np.array(D_OUT)
# Measure Variation
VAR_IN = D_IN[:-1]-D_IN[1:]
VAR_OUT = D_OUT[:-1]-D_OUT[1:]
# last Reshape in 2D
D_IN_END = D_IN[-1].reshape(5,5)
D_OUT_END = D_OUT[-1].reshape(3,3)
## Frequency
F_IN, F_OUT = [], []
for i in range(NB_GEN) :
    F_IN += [sp.fft.fft(D_IN[i][None])]
    F_OUT += [sp.fft.fft(D_OUT[i][None])]
F_IN, F_OUT = np.concatenate(F_IN), np.concatenate(F_OUT)
# modolu square
MOD_IN = np.real(F_IN*F_IN.conjugate())
MOD_OUT = np.real(F_OUT*F_OUT.conjugate())

## Figure
fig = plt.figure(figsize=(5,3), dpi=240)
ax = fig.add_subplot(111)

#plt.ylabel('Y'); plt.xlabel('X')
plt.ylabel('Generation')#; plt.xlabel('XY Projection')

#im = plt.imshow(D_IN_END, origin='lower', cmap='Greys', aspect="auto")
#im = plt.imshow(D_OUT,  cmap='Greys', origin='lower', extent=[-np.pi,np.pi,1,NB_GEN], aspect="auto")
im = plt.imshow(abs(F_IN),  cmap='Greys', origin='lower', aspect="auto")

plt.colorbar(im)

plt.savefig("OUT/DENSITY.png", dpi=600)
plt.show(); plt.close()


### ADJACENCY MAP OF AGENT MAPPING

fig = plt.figure(figsize=(NB_BEST, NB_GEN))
# grid plot
grid = plt.GridSpec(NB_GEN, NB_BEST, hspace=0.1, wspace=0.1)
# subplot
ax = [[fig.add_subplot(grid[i, j], xticklabels=[], yticklabels=[]) for i in range(NB_GEN)] for j in range(NB_BEST)]
ax = list(zip(*ax))
# loop gen
I = 0
for i, GG in GB_GEN :
    print(str(I)+'/'+str(NB_GEN))
    # Ranking 
    GGS = GG.sort_values('RANK')
    ## Extract neuron per gen
    NEURON = GGS['NEURON_LIST']
    # loop n-plicat
    J = 0
    for N in NEURON[:NB_BEST]:
        # Extract adjacency map
        pos, G = EF.NEURON_2_GRAPH(N)
        ADJ_M = nx.convert_matrix.to_numpy_array(G)
        # image
        plt.axis('on')
        ax[I][J].imshow(ADJ_M, cmap='Greys', vmax = 2)
        ax[I][J].set_aspect('equal')
        J+=1
    I+=1
plt.savefig("OUT/ADJ.png", dpi=360)
plt.show(); plt.close()

### GRAPH DISTRIBUTION CONSTRUCTION
DF_ALL_GRAPH = pd.DataFrame(columns=['GEN','RANK','NODE','BETWEENNESS','CLOSENESS','DEGREE','EIGEN'])
for i, d in DF.iterrows() :
    # Extract Graph
    pos, G = EF.NEURON_2_GRAPH(d['NEURON_LIST'])
    
    # BCDE Parameter
    betweenness = nx.betweenness_centrality(G)
    DF_B = pd.DataFrame(betweenness.items(), columns=['NODE','BETWEENNESS'])
    
    closeness = nx.closeness_centrality(G)
    DF_C = pd.DataFrame(closeness.items(), columns=['NODE','CLOSENESS'])['CLOSENESS']
    
    degree = nx.degree_centrality(G)
    DF_D = pd.DataFrame(degree.items(), columns=['NODE','DEGREE'])['DEGREE']
    
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    DF_E = pd.DataFrame(eigenvector.items(), columns=['NODE','EIGEN'])['EIGEN']

    DF_GRAPH = pd.concat([DF_B, DF_C, DF_D, DF_E], axis=1)
    
    # Add gen info
    DF_GRAPH['GEN'] = pd.Series(d['GEN']*np.ones(len(pos)), dtype=int)
    # Add rank info
    DF_GRAPH['RANK'] = pd.Series(d['RANK']*np.ones(len(pos)), dtype=int)
    # Append graph
    DF_ALL_GRAPH = DF_ALL_GRAPH.append(DF_GRAPH, ignore_index=True)

### BCDE Evolution parameter
# Generation statistic's
GRAPH_GB_GEN = DF_ALL_GRAPH.groupby('GEN')
# Loop
p = 'EIGEN' # 'BETWEENNESS', 'CLOSENESS','DEGREE', 'EIGEN']
LOSER, BEST, ALL = [], [], []
for i, GGG in GRAPH_GB_GEN :
    # loser, best dataframe pointer
    GGG_LOSER = GGG[GGG.RANK > (NB_BEST+1)*NB_BEST]
    GGG_BEST = GGG[GGG.RANK < NB_BEST+1]
    # median, std
    MEDIAN, STD = GGG.median(), GGG.std()
    MED_L, STD_L = GGG_LOSER.median(), GGG_LOSER.std()
    MED_B, STD_B = GGG_BEST.median(), GGG_BEST.std()
    # extract parameter
    LOSER += [[MED_L[p], STD_L[p]]]
    BEST += [[MED_B[p], STD_B[p]]]
    ALL += [[MEDIAN[p], STD[p]]]
# to array
LOSER, BEST, ALL = np.array(LOSER), np.array(BEST), np.array(ALL)
# Figures
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)

plt.ylabel(p); plt.xlabel('Generation')
# time abscisse
t = np.arange(0,NB_GEN)
for d,l in [[LOSER,'MIN'],[BEST,'MAX'],[ALL,'ALL']] :
    y_, y_err = d[:,0], d[:,1]
    ax.plot(t, y_, 'o--', label=l)
    #ax.fill_between(t, y_ - y_err, y_ + y_err, alpha=0.2)

plt.legend()
plt.tight_layout() # no label outside

plt.savefig("OUT/GRAPH_TIME.png", dpi=360)
plt.show(); plt.close()

### Phylogenetic Tree
## Construct Tree
TREE = DF['TREE']
pos, G = EF.LINEAGE_2_GRAPH(NB_GEN,NB_P_GEN,TREE)
N_AGENT_TOT = NB_GEN*NB_P_GEN
## Edges Flow
# shortest path
SHORT_PATH = nx.shortest_path(G)
# Construct flow (for each edges beetween 2 nodes, add +1 if exist)
for source in range(N_AGENT_TOT) :
    # listing of each link
    for target, target_list in SHORT_PATH[source].items() :
        # add +1 weight
        if target != source :
            for i in range(len(target_list)-1):
                s,t = target_list[i:i+2]
                G[s][t]['weight'] += 1
# extract edges values
edges_size = np.array([G[u][v]['weight'] for u,v in G.edges]) 
node_size = np.array([len(values) for key,values in SHORT_PATH.items()])+10
# Adding score color
SCORE = DF['SCORE'].values
nodes_color= (SCORE-SCORE.min())/(SCORE.max()-SCORE.min())
# Draw Tree graph
nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.5, node_color=nodes_color)
edges = nx.draw_networkx_edges(G, pos, alpha=2./3, edge_color=edges_size, edge_cmap=plt.cm.Blues)
plt.savefig("OUT/Tree.png", dpi=360)
plt.savefig("OUT/Tree.svg")
plt.show(); plt.close()

## Dispersion of Agent
# extract ancestor of last gen
ANCESTOR = []
N_LAST = 2 # 0 if the real last
for i in range(N_AGENT_TOT-(N_LAST+1)*NB_P_GEN,N_AGENT_TOT-N_LAST*NB_P_GEN,) :
    ANCESTOR += [[i, nx.ancestors(G,i)]]
ANCESTOR = np.array(ANCESTOR)
# select best ancestor
BEST_A = []
for a in ANCESTOR:
    if len(a[1]) != 0 :
        GRANDPA = min(list(a[1]))
    else :
        GRANDPA = a[0]
    BEST_A += [[GRANDPA, len(a[1])]]
BEST_A = np.array(BEST_A)
# find better
TRI_ANCESTOR = np.unique(BEST_A, axis=0)
TRI_ANCESTOR = TRI_ANCESTOR[np.argsort(TRI_ANCESTOR[:,1])[::-1]]


# id Trajectory
NB_ORDER = 2
idx_list = np.where((BEST_A == TRI_ANCESTOR[NB_ORDER]).all(axis=1))[0]
ID_ =  ANCESTOR[idx_list][:,0].astype(int)
ID_BOOL = np.full(N_AGENT_TOT, False)
ID_BOOL[ID_] = True
# extraction loop
GRAD_MVT = []
for i, d in DF[ID_BOOL].iterrows() :
    XY_AGT = np.array(d['AGENT_POS'])
    GRAD = XY_AGT[1:]-XY_AGT[:-1]
    # delete outlier
    NORM = np.linalg.norm(GRAD, axis=1)
    # add all mvt
    GRAD_MVT += [GRAD[NORM < 2]]
# concatenate
GRAD_MVT = np.concatenate(GRAD_MVT)
# angles
THETA = np.arctan2(GRAD_MVT[:,0],GRAD_MVT[:,1])
# Histogram
fig = plt.figure(figsize=(5,5), dpi=240)
ax = fig.add_subplot(111)

plt.ylabel('Number (normed)'); plt.xlabel('Angles (rad)')
plt.xlim(0,np.pi)

plt.hist(THETA, density=True)

plt.savefig("OUT/TRAJ_HIST.png", dpi=600)
plt.show(); plt.close()

"""






"""

### SCORE DISTRIBUTION
GB_GEN = DF.groupby('GEN')
SCORE = GB_GEN['SCORE']

DATA_SCORE = [s.values for i,s in SCORE]

plt.xlabel('Generation'); plt.ylabel('SCORE (distribution)')
plt.violinplot(DATA_SCORE)
#plt.savefig("OUT/SCORE.png", dpi=600)
plt.show(); plt.close()


### TRAJECTORY DISTRIBUTION
DF_GEN_ONE = DF[DF['GEN'] == 0]
GRAD = []
for i, d in DF_GEN_ONE.iterrows() :
    # extract pos
    XY_AGT = np.array(d['AGENT_POS'])
    # gradiant (=vector)
    V = XY_AGT[1:]-XY_AGT[:-1]
    # list
    GRAD += [V]
GRAD = np.concatenate(GRAD)
THETA = np.arctan2(GRAD[:,0],GRAD[:,1])
# histogram
N = 20
HIST = np.histogram(THETA, bins=N)
ax = plt.axes([0.025, 0.025, 0.95, 0.95], polar=True)

theta = np.arange(0.0, 2 * np.pi, 2 * np.pi / N)
width = (2 * np.pi / N) * np.ones(N)
bars = plt.bar(theta, HIST[0], width=width, bottom=0.0, alpha=0.9)

#plt.savefig("OUT/TRAJECTORY.png", dpi=600)
plt.show(); plt.close()

"""