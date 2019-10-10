import numpy as np
from gym import Env, spaces
import math


class Simulator(Env):
    def __init__(self):
        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.initial_state = np.array([9.1, 6.8, 9.1, 6.8, 6.8, 6.8])
        self.relative_volatility = np.array([3.5, 1.2, 2.7, 1.21, 3.0])  # A/B, B/C etc
        discrete_action_size = self.initial_state.shape[0] - 1  # action selects LK
        continuous_action_number = 1
        self.state = self.initial_state.copy()
        self.max_columns = 10

        # spaces for mixed action space?
        self.discrete_action_space = spaces.Discrete(discrete_action_size)
        self.continuous_action_space = spaces.Box(low=0.8, high=0.999, shape=(continuous_action_number,))
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

    def step(self, action):
        # note that same_action_punish should get removed as it is a hard coded heuristic
        # TODO rewrite this without all the silly constraints
        reward = 0
        action_continuous, action_discrete = action
        LK_split = self.action_continuous_definer(action_continuous)
        self.split_order.append(LK_split)
        Light_Key = action_discrete
        self.sep_order.append(Light_Key)

        done = False
        self.steps += 1
        if self.steps > 20:  # episode ends after 20 distillation columns
            done = True
        previous_state = self.state.copy()


        Heavy_Key = Light_Key + 1
        HK_split = 1 - LK_split
        # HK_split = action[2]
        tops = np.zeros(self.initial_state.shape)
        tops[:Light_Key + 1] = self.state[:Light_Key + 1]
        tops[Light_Key] = tops[Light_Key] * LK_split
        tops[Heavy_Key] = previous_state[Heavy_Key] * HK_split
        bots = previous_state - tops
        if sum(tops) == 0 or sum(bots) == 0:
            reward = -50
            return self.state, reward, done, {}

        LK_D = tops[Light_Key] / sum(tops)
        LK_B = bots[Light_Key] / sum(bots)
        if LK_D in [0,1] or LK_B in [0, 1]:
            return self.state, reward, done, {}
        self.stream_table.append(tops)
        self.stream_table.append(bots)

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
            # reward proportional to stream flow and purity^2
            reward += self.state.sum()*max(np.divide(self.state, self.state.sum()))**2
            if np.array_equal(self.state, self.stream_table[-1]):
                done = True
                break
            self.current_stream += 1
            self.state = self.stream_table[self.current_stream]

        if np.isnan(reward) or np.isinf(reward):
            reward = -50
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
            LK = np.random.randint(0, self.initial_state.size - 1)
            LK_split = np.random.rand(1)
            action = np.array([LK, LK_split])
            state, reward, done, _ = self.step(action)
            print(f'reward: {reward}, LK: {LK}, LK_split: {LK_split}')

    def action_continuous_definer(self, action_continuous):
        # agent gives continuous argument between -1 and 1 (width 2)
        # reformat split agent action * split range / agent action range + (split minimum - agent minimum)
        LK_Split = self.continuous_action_space.low + (action_continuous - (-1)) / 2 \
                   * (self.continuous_action_space.high - self.continuous_action_space.low)
        return LK_Split
