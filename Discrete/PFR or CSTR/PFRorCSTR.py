# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:18:00 2019

@author: meatrobot
"""
import numpy as np
from gym import Env, spaces
import matplotlib.pyplot as plt
import matplotlib.animation as FuncAnimation
from matplotlib import style




class simulator(Env):
    def __init__(self, n_reactors=9, action_size=2, state_size=1, volume=1):
        self.volume = volume
        self.reactor_seq = []
        self.X = [0]
        self.state = None
        #self.action_size = action_size
        #self.state_size = n_reactors + 1 #1 for "done" for discrete state
        #self.state_size = state_size
        self.n_reactors = n_reactors
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low=0, high=1, shape = (1,))
        self.n_steps = 0



    def choose_reactor(self, action):
        self.reactor_seq.append(action)
        
    def step(self, action):
        self.n_steps += 1
        self.choose_reactor(action)
        self.X.append(self.equation_solver(action, self.X[-1]))
        self.state = np.array([self.X[-1]])  # seems like most envs give state as a np array
        reward = self.X[-1] - self.X[-2]  #increase in conversion
        if self.n_steps == self.n_reactors or self.X[-1] >= 1:
            done = True
        else: 
            done = False
        return self.state, reward, done, {}
    
    def show_seq(self):
        return self.reactor_seq, self.X
        
    
    def reset(self):
        self.n_steps = 0
        self.reactor_seq = [] 
        self.X = [0]
        self.state = np.array([0])
        return self.state
    
    def render(self, mode='human'):
        print(f'choice({len(self.X)-1}) - conversions: {self.X}, reactors: {self.reactor_seq}')

    def equation_solver(self, r_type, X_prev, a=-10, b=10, c=2):
        Vol = self.volume
        X_new = X_prev
        # see page 13 of report
        if r_type == 0: # CSTR
            A = a
            B = b - a*X_prev
            C = c - b*X_prev
            D = -c*X_prev - Vol
            coeffs = [A, B, C, D]
        else:
            A = a/3
            B = b/2
            C = c
            D = -(a/3 * X_prev**3 + b/2*X_prev**2 + c* X_prev) - Vol
            coeffs = [A, B, C, D]
            
        roots = np.roots(coeffs)
            
        if True in np.isreal(roots):
            roots = roots[np.isreal(roots)]
            pos_roots = np.array([root for root in roots if root>X_prev])
            if pos_roots.size > 0:
                diffs = pos_roots - X_prev  #closest conversion solution above X prev
                X_new = pos_roots[diffs == (min(diffs))][0]
            else:
                X_new = 1
        if X_new > 1: 
            X_new = 1
            
        return X_new
