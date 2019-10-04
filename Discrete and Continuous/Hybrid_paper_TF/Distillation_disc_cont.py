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
"""In general, without just using a big punishment, 
how does one limit a NN to make choices that make sense with prior knowledge"""

    
class Simulator(Env):
    def __init__(self):
        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.initial_state = np.array([9.1, 6.8, 9.1, 6.8, 6.8, 6.8])
        self.relative_volatility = np.array([3.5, 1.2, 2.7, 1.21, 3.0]) #A/B, B/C etc
        discrete_action_size = self.initial_state.shape[0] - 1  #action selects LK
        continuous_action_number = 1 
        self.state = self.initial_state.copy()
        self.max_columns = 10

        #spaces for mixed action space?
        self.discrete_action_space = spaces.Discrete(discrete_action_size)
        self.continuous_action_space = spaces.Box(low=0.8, high=0.999, shape=(continuous_action_number,))
        self.observation_space = spaces.Box(low = 0, high = 9.1, shape = self.initial_state.shape)
        
        self.total_cost = 0
        self.stream_table = [self.initial_state.copy()]
        self.outlet_streams = []
        self.sep_order = []
        self.split_order = []
        self.current_stream = 0
        self.steps = 0
        
        empty = np.zeros(self.initial_state.shape)
        def maker(empty, initial_state, i): 
            stream = empty.copy()
            stream[i] = initial_state[i]
            return stream
        self.product_streams = [maker(empty, self.initial_state, i) for i in range(self.initial_state.size)]
        
    def step(self, action, same_action_punish=True): #note that same_action_punish should get removed as it is a hard coded heuristic
        reward = 0
        action_continuous, action_discrete = action
        LK_split= self.action_continuous_definer(action_continuous)
        done = False
        self.steps += 1
        if self.steps > 20: done = True
        previous_state = self.state.copy()       
        Light_Key = action_discrete
        self.sep_order.append(Light_Key)
        Heavy_Key = Light_Key + 1
        self.split_order.append(LK_split)
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
            reward = -100 #big punishment, and state etc remain the same
        else:                #valid action
            if len(self.sep_order) > self.max_columns: done = True
            self.stream_table.append(tops)
            self.stream_table.append(bots)
            N =  np.log(LK_D/(1-LK_D) * (1-LK_B)/LK_B)/np.log(self.relative_volatility[Light_Key])
            Cost = abs(N)
            if math.isnan(Cost): Cost = 100   #check how it's possible that cost can be negative?
            self.total_cost += N
            reward += -Cost

            if len(self.sep_order) > 1:
                if Light_Key == self.sep_order[-2] and same_action_punish: #repeating actions is bad
                    reward = -100
                if Light_Key != self.sep_order[-2]: #action can't be a repeat to get reward for making a product stream
                    #if tops or bottoms are product stream reward +=100
                    if min(np.sum(abs(self.product_streams - tops), axis=0)) < 0.1:
                        reward += 50
                    if min(np.sum(abs(self.product_streams - bots), axis=0)) < 0.1:
                        reward += 50
                    
            """Go to next stream as state, if stream only contains more than 95%wt of a single compound then go to next stream"""
            self.current_stream +=1
            self.state = self.stream_table[self.current_stream]
            while max(np.divide(self.state,self.state.sum())) > 0.95: 
                self.outlet_streams.append(self.state)
                reward += self.state.sum()**2*max(np.divide(self.state,self.state.sum()))**2 #reward proportional to stream flow and purity
                if  np.array_equal(self.state,self.stream_table[-1]):
                    done = True
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
        self.steps = 0
        self.outlet_streams = []
        self.split_order = []
        return self.state
    
    def render(self, mode='human'):
        print(f'total cost: {self.total_cost} sep_order: {self.sep_order} split_order: {self.split_order} \n')
    
    def test_random(self, n_steps=5):
        for i in range(n_steps): 
            LK = np.random.randint(0, self.initial_state.size-1)
            LK_split = np.random.rand(1)
            action = np.array([LK, LK_split])
            state, reward, done, _ = self.step(action)
            print(f'reward: {reward}, LK: {LK}, LK_split: {LK_split}')

    def action_continuous_definer(self, action_continuous):
        # agent gives continuous argument between -1 and 1 (width 2)
        # reformat split agent action * split range / agent action range + (split minimum - agent minimum)
        LK_Split = action_continuous*(self.continuous_action_space.high - self.continuous_action_space.low)/2 \
                   + self.continuous_action_space.low - (-1)
        return LK_Split
