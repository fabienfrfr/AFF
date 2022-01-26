#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:40:35 2022
@author: fabien
"""

import gym
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from collections import namedtuple

import FUNCTIONNAL_FILLET_ as FF

##################################### FUNCTION
## For Q-Learning
def Q_TABLE(model, batch, GAMMA = 0.9):
    old_state = torch.tensor(np.concatenate(batch.state), dtype=torch.float)
    action = torch.tensor(np.array(batch.action), dtype=torch.long).unsqueeze(1)
    new_state = torch.tensor(np.concatenate(batch.next_state), dtype=torch.float)
    reward = torch.tensor(np.array(batch.reward), dtype=torch.long)
    done = torch.tensor(np.array(batch.done), dtype=torch.int)
    # actor proba
    actor = model(old_state)
    # Compute predicted Q-values for each action
    pred_q_values_batch = actor.gather(1, action)
    pred_q_values_next  = model(new_state)
    # Compute targeted Q-value for action performed
    target_q_values_batch = (reward+(1-done)*GAMMA*torch.max(pred_q_values_next, 1)[0]).detach().unsqueeze(1)
    # return y, y_prev
    return pred_q_values_batch,target_q_values_batch

## Accuracy variable construction factor
def Factor_construction(df, main_group, main_value, sub_group, factor):
    # extract data
    sub_df = df[df[main_group] == main_value]
    fact_abs = sub_df.groupby(sub_group)[factor].mean().values
    fact_rel = (fact_abs-fact_abs.min())/(fact_abs.max()-fact_abs.min())
    return fact_rel

def FIT(env,MODEL):
    # loop
    duration = pd.DataFrame(columns=['GEN','IDX_SEED', 'EPISOD', 'DURATION']).astype(int)
    for g in tqdm(range(MODEL.NB_GEN)) :
        # per seeder
        for n in range(MODEL.NB_SEEDER):
            # train
            render,l = 0,0
            for i in range(MODEL.NB_E_P_G):
                new_state = env.reset()
                done = False
                # gen data
                j = 0
                while not done :
                    action = MODEL.STEP(new_state[None], n)
                    state = new_state
                    new_state, reward, done, info = env.step(action)
                    if done and j < N_MAX-10 :
                        reward = -10
                    # see
                    if render == 0 :
                        env.render()
                    MODEL.memory[n].push(state[None], action, new_state[None], reward, done)
                    # iteration
                    j+=1
                    l+=1
                # duration
                duration = duration.append({'GEN':g, 'IDX_SEED':n,'EPISOD':i,'DURATION':j},ignore_index=True)
                render+=1
                # fit
                if l >= MODEL.BATCH :
                    nb_batch = min(int(MODEL.memory[n].__len__()/MODEL.BATCH), np.rint(j/MODEL.BATCH).astype(int))
                    transitions = MODEL.memory[n].sample(nb_batch*MODEL.BATCH)
                    # batch adapted loop
                    for b in range(nb_batch) :
                        batch = Transition(*zip(*transitions[b*MODEL.BATCH : (b+1)*MODEL.BATCH]))
                        pred_q, target_q = Q_TABLE(MODEL.SEEDER_LIST[n], batch)
                        MODEL.TRAIN(pred_q, target_q, g, n, i, b)
        # Accuracy contruction
        duration_factor = Factor_construction(duration, 'GEN', g, 'IDX_SEED', 'DURATION')
        duration_factor = 1 - duration_factor # for odering
        # Apply natural selection
        MODEL.SELECTION(g, supp_factor=duration_factor)
        #if g == 1 : break
    ## Finalization
    MODEL.FINALIZATION(supp_param=duration,save=True)
    env.close()
    return duration

##################################### ALGO

if __name__ == '__main__' :
    LOAD = True
    Filename = 'MODEL_FF_20220125_172002.obj' #25 : 5h
    ## env gym
    env = gym.make("CartPole-v0")
    N_MAX = 300
    env._max_episode_steps=N_MAX
    NB_OBS = env.observation_space.shape[0]
    NB_ACTION = env.action_space.n
    
    ## parameter
    IO =  (NB_OBS,NB_ACTION)
    BATCH = 25
    NB_GEN = 100
    NB_SEED = 5**2
    NB_EPISODE = 25000 #25000
    
    ## Load previous model or launch new
    if LOAD :
        Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done')) #important
        MODEL = FF.FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE], Transition, TYPE='RL')
        MODEL = MODEL.LOAD(Filename)
    else :
        Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))
        MODEL = FF.FunctionnalFillet([IO, BATCH, NB_GEN, NB_SEED, NB_EPISODE], Transition, TYPE='RL')
        # Fit
        FIT(env, MODEL)
    
    ## extract some variable
    duration = MODEL.supp_param
    print(MODEL.PARENTING)
    
    ### plot
    import pylab as plt
    from scipy.ndimage import filters
    
    ## 2 simple curve
    ctrl = duration[duration.IDX_SEED==0].groupby('GEN')
    evol = duration[duration.IDX_SEED!=0].groupby('GEN')
    
    score, std, min_, max_ = [], [], [], []
    for c in [ctrl,evol] :
        score += [np.squeeze(c.agg({'DURATION':'max'}).values)] 
        std += [np.squeeze(c.agg({'DURATION':'std'}).values)] 
        min_ += [score[-1].min()]
        max_ += [score[-1].max()]
    min_, max_ = min(min_), max(max_)
    
    W, H, L, S = 3.7, 3.7, 18., 9. # width, height, label_size, scale_size
    #W, H, L, S = 3.7, 2.9, 18., 9. # width, height, label_size, scale_size
    # fig ratio
    MM2INCH = 1# 2.54
    W, H, L, S = np.array((W, H, L, S))/MM2INCH # ratio fig : 2.7/2.1
    STD = 1
    # Figure
    fig = plt.figure(figsize=(W, H))
    
    plt.rc('font', size=S)
    plt.rc('axes', titlesize=S)
    
    ax = fig.add_subplot()
    ax.set_title('CartPole-v1', fontsize=L)
    ax.set_ylabel('Time (relative)', fontsize=L)
    ax.set_xlabel('GEN', fontsize=L)
    x_reduce = np.arange(len(score[0]))
    for z in zip(score,std):
        curve = filters.gaussian_filter1d((z[0]-min_)/(max_-min_),1)
        inter = filters.gaussian_filter1d((z[1])/(max_-min_),1)
        
        ax.plot(curve)
        ax.fill_between(x_reduce, curve - inter/STD, curve + inter/STD, alpha=0.3)
    
    plt.xlim([0,100])
    plt.ylim([0,0.9])
    
    # Save data
    import os
    plt.savefig('OUT' + os.path.sep + 'CartPole-v1_' + 'b_' +str(BATCH)+"n_"+str(100)+".svg")
    plt.show(); plt.close()
    """
    duration_sep = duration.groupby(['IDX_SEED','GEN']).agg({"DURATION":"mean"})
    plt.plot(duration_sep)
    """
