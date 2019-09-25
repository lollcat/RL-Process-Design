# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:18:00 2019

@author: meatrobot
"""
import numpy as np
from gym import Env, spaces
import math


    
class simulator(Env):
    def __init__(self):
        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.initial_state = np.array([9.1, 6.8, 9.1, 6.8, 6.8, 6.8])
        self.relative_volatility = np.array([3.5, 1.2, 2.7, 1.21, 3.0]) #A/B, B/C etc
        action_size = self.initial_state.shape[0] - 1 #action selects LK
        self.state = self.initial_state.copy()
        self.max_columns = 10
        
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low = 0, high = 9.1, shape = self.initial_state.shape)      
        
        self.total_cost = 0
        self.stream_table = [self.initial_state.copy()]
        self.outlet_streams = []
        self.sep_order = []
        self.current_stream = 0
        self.silly_counter = 0
        self.LK_split = 0.99
        self.HK_split = (1-0.99)
        
        empty = np.zeros(self.initial_state.shape)
        def maker(empty, initial_state, i): 
            stream = empty.copy()
            stream[i] = initial_state[i]
            return stream
        self.product_streams = [maker(empty, self.initial_state, i) for i in range(self.initial_state.size)]
        
    def step(self, action):
        done = False
        previous_state = self.state.copy()
        tops = np.zeros(self.initial_state.shape)
        tops[:action+1] = self.state[:action+1]
        tops[action] = tops[action]*self.LK_split 
        tops[action+1] = previous_state[action+1]*self.HK_split
        bots = previous_state - tops
        LK_D = tops[action]/sum(tops)
        LK_B = bots[action]/sum(bots)
        #print(LK_D)
        #print(LK_B)
        if LK_D in [1,0] or LK_B in [1,0] or math.isnan(LK_D) or math.isnan(LK_B): #invalid action (HK LK split doesnt exist)
            reward = -1000 #big punishment, and state etc remain the same
            self.silly_counter +=1
            if self.silly_counter >= 10:
                done = True
        else:                #valid action  
            self.sep_order.append(action)
            if len(self.sep_order) > self.max_columns: done = True
            self.stream_table.append(tops)
            self.stream_table.append(bots)
            N =  np.log(LK_D/(1-LK_D) * (1-LK_B)/LK_B)/np.log(self.relative_volatility[action])
            Cost = N*1
            if Cost < 0: Cost = 1000
            self.total_cost += N
            reward = -Cost
            if len(self.sep_order) > 1:
                if action != self.sep_order[-2]: #action can't be a repeat
                    #if tops or bottoms are product stream reward +=100
                    if min(np.sum(abs(self.product_streams - tops), axis=0)) < 0.2:
                        reward += 100
                    if min(np.sum(abs(self.product_streams - bots), axis=0)) < 0.2:
                        reward +=100
                    
            """Go to next stream as state, if stream only contains 1 compound then go to next stream"""
            self.current_stream +=1
            self.state = self.stream_table[self.current_stream]
            while sum(np.divide(self.state,self.initial_state)) < 1.5: 
                self.outlet_streams.append(self.state)
                if  np.array_equal(self.state,self.stream_table[-1]):
                    done = True
                    reward = 1000 # reward for finishing
                    break 
                self.current_stream +=1
                self.state = self.stream_table[self.current_stream]
        return self.state, reward, done, {}
    
    def reset(self): 
        self.state = self.initial_state.copy()
        self.stream_table = [self.initial_state.copy()]
        self.current_stream = 0
        self.sep_order = []
        self.total_cost = 0
        return self.state
    
    def render(self, mode = 'human'):
        print(f'total cost: {self.total_cost} sep_order = {self.sep_order}')
    
    """def test(self):
        print("all good")"""

"""
env = simulator()
for i in range(5):   
    state, reward, done, _ = env.step(i)
    print(reward)


env = simulator()
for i in range(5):   
    state, reward, done, _ = env.step(1)
    print(reward)
"""