# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:18:00 2019

@author: meatrobot
"""
import numpy as np
from gym import Env, spaces


    
class simulator(Env):
    def __init__(self, n_units=9, action_size = 3, state_size = 2, Tmax = 600):
        self.Tmax = Tmax
        self.unit_seq = []
        self.X = [0]
        self.T = [Tmax]
        self.state = None
        self.n_units = n_units
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low = np.array([0,0]), high = np.array([1, Tmax]))
     
    def update_sequence(self, action):
        if action == 0: self.unit_seq.append("CSTR")
        elif action == 1: self.unit_seq.append("PFR")
        elif action == 2: self.unit_seq.append("Cooler")
    
    def step(self, action):
        self.update_sequence(action)
        self.simulate(action)
        Xnew, Tnew = self.state
        self.X.append(Xnew)
        self.T.append(Tnew)
        reward = self.X[-1] - self.X[-2]  #increase in conversion
        if len(self.unit_seq)> self.n_units-1 or self.X[-1] >= 1:
            done = True
        else: 
            done = False
        return self.state, reward, done, {}
    
    def reset(self): 
        self.unit_seq = [] 
        self.X = [0]
        self.T = [self.Tmax]
        self.state = np.array([0, self.Tmax])
        return self.state
    
    def render(self, mode = 'human'):
        print(f'choice({len(self.X)-1}) -  units: {self.unit_seq}, conversions: {self.X}, Temps: {self.T}')
        
    def simulate(self, action, Vol = 1, a = -10, b = 10, c=2):
        X_prev = self.X[-1]
        T_prev = self.T[-1]
        if action in [0,1]: #reactor
            if action == 0: # CSTR
                T_new = T_prev - 20
                A = a
                B = b + X_prev
                C = c - b*X_prev
                D = -c*X_prev - Vol
                coeffs = [A, B, C, D]
            else: #PFR
                T_new = T_prev - 20
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
                    diffs =  pos_roots - X_prev  #closest conversion solution above X prev
                    X_new = pos_roots[diffs == (min(diffs))][0]
                else: X_new = X_prev
            else: X_new = 1
            X_new = X_prev + max(0,(X_new - X_prev)*(1 - (self.Tmax - T_new)**2/4000))
            if X_new > 1:
                X_new = 1    
            self.state = np.array([X_new, T_new])         
        elif action == 2:
            self.state = np.array([X_prev, self.Tmax])
            
