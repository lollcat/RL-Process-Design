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
    def __init__(self, split_option_n=3):
        # discretised continuous split space
        self.split_option_n = split_option_n
        self.split_max = 0.99
        self.split_min = 0.9
        self.split_options = np.linspace(self.split_min, self.split_max, num=self.split_option_n)

        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.initial_state = np.array([9.1, 6.8, 9.1, 6.8, 6.8, 6.8])
        self.relative_volatility = np.array([3.5, 1.2, 2.7, 1.21, 3.0])  # A/B, B/C etc
        discrete_action_size = (self.initial_state.shape[0] - 1) * self.split_option_n  # action selects LK and split
        self.state = self.initial_state.copy()
        self.max_columns = 10

        self.action_space = spaces.Discrete(discrete_action_size)
        self.observation_space = spaces.Box(low=0, high=9.1, shape=self.initial_state.shape)
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
        
    def step(self, action, same_action_punish=True):
        reward = 0

        # Get splits and light key values
        Light_Key = int(action / self.split_option_n)
        LK_split_number = action % self.split_option_n
        LK_split = self.split_options[LK_split_number]
        self.split_order.append(LK_split)
        self.sep_order.append(Light_Key)

        done = False
        self.steps += 1
        if self.steps > self.max_columns:
            done = True  # TODO add this change to disc_cont environment
        previous_state = self.state.copy()

        # calculate tops and bottoms flows and add to stream table
        Heavy_Key = Light_Key + 1
        HK_split = 1 - LK_split
        tops = np.zeros(self.initial_state.shape)
        tops[:Light_Key] = self.state[:Light_Key]
        tops[Light_Key] = previous_state[Light_Key]*LK_split
        tops[Heavy_Key] = previous_state[Heavy_Key]*HK_split
        bots = previous_state - tops
        LK_D = tops[Light_Key]/sum(tops)
        LK_B = bots[Light_Key]/sum(bots)
        self.stream_table.append(tops)
        self.stream_table.append(bots)

        if len(self.sep_order) > self.max_columns:
            done = True

        # calculate number of stages using the fenske equation & give punishment
        n_stages = np.log(LK_D/(1-LK_D) * (1-LK_B)/LK_B)/np.log(self.relative_volatility[Light_Key])
        cost = n_stages
        self.total_cost += cost
        reward += -cost

        # if tops or bottoms are product stream reward +=10
        if min(np.sum(abs(self.product_streams - tops), axis=0)) < 0.1:
            reward += 10
        if min(np.sum(abs(self.product_streams - bots), axis=0)) < 0.1:
            reward += 10

        # Go to next stream as state, if stream only contains more than 0.9 wt% of a single compound
        # then go to next stream
        self.current_stream += 1
        self.state = self.stream_table[self.current_stream]
        while max(np.divide(self.state, self.state.sum())) > 0.9:
            self.outlet_streams.append(self.state)
            # reward proportional to stream flow and purity
            reward += self.state.sum()*max(np.divide(self.state, self.state.sum()))**2
            if np.array_equal(self.state, self.stream_table[-1]):
                done = True
                break
            self.current_stream += 1
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
            LK_split = np.random.choice(self.split_options)
            action = LK * self.split_option_n + LK_split
            state, reward, done, _ = self.step(action)
            print(f'reward: {reward}, LK: {LK}, LK_split: {LK_split}')
