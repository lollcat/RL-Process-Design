# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:18:00 2019

@author: meatrobot
"""
import numpy as np
from gym import Env, spaces
import math


"""Actions: choose LK and LK split""" 
"""HK split has to be less than LK split - how to add constraint?"""
"""In general, without just using a big punishment, how does one limit a NN to make choices that make sense with prior knowledge"""

    
class simulator(Env):
    def __init__(self):
        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.initial_state = np.array([9.1, 6.8, 9.1, 6.8, 6.8, 6.8])
        self.relative_volatility = np.array([3.5, 1.2, 2.7, 1.21, 3.0]) #A/B, B/C etc
        discrete_action_size = self.initial_state.shape[0] - 1 #action selects LK
        continuous_action_number = 1 
        self.state = self.initial_state.copy()
        self.max_columns = 10
        
        #spaces for mixed action space?
        self.discrete_action_space = spaces.Discrete(discrete_action_size)
        self.continuous_action_space = spaces.Box(low = 0.5, high = 1, shape = (continuous_action_number,))
        self.observation_space = spaces.Box(low = 0, high = 9.1, shape = self.initial_state.shape)
        
        self.total_cost = 0
        self.stream_table = [self.initial_state.copy()]
        self.outlet_streams = []
        self.sep_order = []
        self.current_stream = 0
        self.silly_counter = 0
        
        empty = np.zeros(self.initial_state.shape)
        def maker(empty, initial_state, i): 
            stream = empty.copy()
            stream[i] = initial_state[i]
            return stream
        self.product_streams = [maker(empty, self.initial_state, i) for i in range(self.initial_state.size)]
        
    def step(self, action):
        action_discrete, action_continuous = action
        done = False
        previous_state = self.state.copy()       
        Light_Key = action_discrete
        Heavy_Key = Light_Key + 1
        LK_split = action_continuous
        HK_split = 1 - LK_split
        #HK_split = action[2]
        tops = np.zeros(self.initial_state.shape)
        tops[:Light_Key+1] = self.state[:Light_Key+1]
        tops[Light_Key] = tops[Light_Key]*LK_split 
        tops[Heavy_Key] = previous_state[Heavy_Key]*HK_split
        bots = previous_state - tops
        LK_D = tops[Light_Key]/sum(tops)
        LK_B = bots[Light_Key]/sum(bots)
        #Gets error for LK_B sometimes if choice doesn't make sense - resolved below
        #print(LK_B)
        if LK_D in [1,0] or LK_B in [1,0] or math.isnan(LK_D) or math.isnan(LK_B): #invalid action (HK LK split doesnt exist)
            reward = -1000 #big punishment, and state etc remain the same
            self.silly_counter +=1
            if self.silly_counter >= 10:
                done = True
        else:                #valid action  
            self.sep_order.append(Light_Key)
            if len(self.sep_order) > self.max_columns: done = True
            self.stream_table.append(tops)
            self.stream_table.append(bots)
            N =  np.log(LK_D/(1-LK_D) * (1-LK_B)/LK_B)/np.log(self.relative_volatility[Light_Key])
            Cost = N*1
            if Cost < 0 or math.isnan(Cost): Cost = 1000   #check how it's possible that cost can be negative?
            self.total_cost += N
            reward = -Cost
            if len(self.sep_order) > 1:
                if Light_Key != self.sep_order[-2]: #action can't be a repeat to get reward
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
                    reward = 100 # reward for finishing
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
        self.silly_counter = 0
        return self.state
    
    def render(self, mode = 'human'):
        print(f'total cost: {self.total_cost} sep_order = {self.sep_order}')
    
    def test_random(self, n_steps = 5):       
        for i in range(n_steps): 
            LK = np.random.randint(0, self.initial_state.size-1)
            LK_split = np.random.rand(1)
            action = np.array([LK, LK_split])
            state, reward, done, _ = self.step(action)
            print(f'reward: {reward}, LK: {LK}, LK_split: {LK_split}')




