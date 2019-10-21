import numpy as np
from gym import Env, spaces
from scipy.optimize import fsolve


class Simulator:
    print("TODO: constrain actions within action select in agent")
    def __init__(self):
        # Compound Data
        self.compound_names = ["Methane", "Ethane", "Propane", "Butane", "Pentane"]
        # , "Hydrogen Sulphide", "Carbon Dioxide", "Nitrogen","Helium"]
        self.initial_state = np.array([77.1, 6.6, 3.1, 2, 3]) # , 3.3, 1.7, 3.2 Flowrates
        # Anotine data from YAWS
        self.Antoine_a1 = np.array([6.84566, 6.95335, 7.01887, 7.00961, 7.00877]) # , 7.11958, 7.58828, 6.72531,5.2712
        self.Antoine_a2 = np.array([435.621, 699.106, 889.864, 1022.48, 1134.15]) # , 802.227, 861.82, 285.573,13.5171
        self.Antoine_a3 = np.array([271.361, 260.264, 257.084, 248.145, 238.678]) # , 249.61, 271.883, 270.087,	274.585

        self.Light_order = np.array([])

        discrete_action_size = self.initial_state.shape[0] - 1  # action selects LK
        continuous_action_number = 1
        self.state = self.initial_state.copy()
        self.max_columns = 10

        # spaces for mixed action space?
        self.discrete_action_space = spaces.Discrete(discrete_action_size)
        self.continuous_action_space_input = spaces.Box(low=-1, high=1, shape=(continuous_action_number,))
        self.continuous_action_space = spaces.Box(low=0.8, high=0.999, shape=(continuous_action_number,))
        self.observation_space = spaces.Box(low=0, high=9.1, shape=self.initial_state.shape)

        self.total_cost = 0
        self.stream_table = [self.initial_state.copy()]
        self.outlet_streams = []
        self.sep_order = []
        self.split_order = []
        self.column_conditions = []  # pressure, tops temperature, bots temperature
        self.column_dimensions = [] # Nstages, Reflux ratio
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
        feed = self.state.copy()


        Heavy_Key = Light_Key + 1
        HK_split = 1 - LK_split
        # HK_split = action[2]
        tops = np.zeros(self.initial_state.shape)
        tops[:Light_Key + 1] = self.state[:Light_Key + 1]
        tops[Light_Key] = tops[Light_Key] * LK_split
        tops[Heavy_Key] = feed[Heavy_Key] * HK_split
        bots = feed - tops
        if sum(tops) == 0 or sum(bots) == 0:
            reward = -50
            return self.state, reward, done, {}

        LK_D = tops[Light_Key] / tops.sum()
        HK_D = tops[Heavy_Key] / tops.sum()
        LK_B = bots[Light_Key] / bots.sum()
        HK_B = bots[Heavy_Key] / bots.sum()

        if LK_D in [0,1] or LK_B in [0, 1]:
            return self.state, reward, done, {}
        self.stream_table.append(tops)
        self.stream_table.append(bots)

        system_pressure, tops_temperature, bots_temperature = self.calculate_conditions(tops, bots)
        self.column_conditions.append([system_pressure, tops_temperature, bots_temperature])

        relative_volatility_top = self.Kvalues_calculator(tops_temperature, system_pressure)[Light_Key] \
                                  / self.Kvalues_calculator(tops_temperature, system_pressure)[Heavy_Key]
        relative_volatility_bot = self.Kvalues_calculator(bots_temperature, system_pressure)[Light_Key] \
                                  / self.Kvalues_calculator(bots_temperature, system_pressure)[Heavy_Key]
        relative_volatility_mean = np.sqrt(relative_volatility_top*relative_volatility_bot)

        # calculate number of stages using the fenske equation & give punishment
        n_stages_min = np.log(LK_D/(HK_D) * HK_B/LK_B)/np.log(relative_volatility_mean)


        # calculate the minimum reflux ratio
        liquidfeed = feed # always completely condensed? # TODO check this
        LK_liquidfeed = liquidfeed[Light_Key]/liquidfeed.sum()
        HK_liquidfeed = liquidfeed[Heavy_Key] / liquidfeed.sum()
        RefluxRatio_min = liquidfeed.sum()/tops.sum() * (LK_D/LK_liquidfeed - relative_volatility_mean * HK_D/HK_liquidfeed) \
                          /(relative_volatility_mean - 1)
        # using mean relative volatility here but should actually be feed relative volatility

        RefluxRatio_actual = 1.3 * RefluxRatio_min

        # Now use Gillibrand correlation to get ideal number of stages
        X = (RefluxRatio_actual - RefluxRatio_min)/(RefluxRatio_actual + 1)
        Y = 1 - np.exp((1 + 54.5* X)/(11 + 117.2 * X)*(X - 1)/ X**0.5)
        gilland_func = lambda N: Y - (N - n_stages_min)/(N+1)
        n_stages_actual = fsolve(gilland_func, n_stages_min)


        self.column_dimensions.append([n_stages_actual, RefluxRatio_actual])


        cost = n_stages_actual
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

        if self.steps > self.max_columns:  # episode ends after 20 distillation columns
            done = True
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
        self.column_conditions = []
        self.column_dimensions = []
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

    def calculate_conditions(self, tops, bots):
        system_pressure = self.condensor_pressure_calculator(tops)
        tops_temperature = self.tops_temperature_calculator(tops, system_pressure)
        bots_temperature = self.bots_temperature_calculator(bots, system_pressure)

        return system_pressure, tops_temperature, bots_temperature

    def condensor_pressure_calculator(self, tops):
        condensor_temperature = 60 # C - so CW can be used
        vapour_pressures = self.vapour_pressure_calculator(condensor_temperature)
        p_bub = np.multiply(vapour_pressures, tops/tops.sum()).sum()
        return p_bub

    def tops_temperature_calculator(self, tops, system_pressure):
        tops_composition = tops/tops.sum()
        temp_guess = np.array([80])
        dew_func = lambda temperature: np.divide(tops_composition,
                                                 self.Kvalues_calculator(temperature, system_pressure)).sum() - 1
        tops_temperature = fsolve(dew_func, temp_guess)

        return tops_temperature

    def bots_temperature_calculator(self, bots, system_pressure):
        bots_composition = bots/bots.sum()
        temp_guess = np.array([80])
        bub_func = lambda temperature: np.multiply(bots_composition,
                                                   self.Kvalues_calculator(temperature, system_pressure)).sum() - 1
        bots_temperature = fsolve(bub_func, temp_guess)

        return bots_temperature

    def vapour_pressure_calculator(self, temperature):
        # Antoine equation. Returns matrix of vapour pressures
        # temperature units are in deg C, vapour pressure needs to be converted from mmHg to kPa
        vapour_pressures = (10**(self.Antoine_a1 - (np.divide(self.Antoine_a2, (self.Antoine_a3 + temperature)))))*0.1333
        return vapour_pressures

    def Kvalues_calculator(self, temperature, system_pressure):
        vapour_pressure = self.vapour_pressure_calculator(temperature)
        Kvalues = vapour_pressure/ system_pressure
        return Kvalues

    def run_random(self):
        action_continuous = self.continuous_action_space_input.sample()
        action_discrete = self.discrete_action_space.sample()
        state, reward, done, _ = self.step([action_continuous, action_discrete])
        return state, reward, done, _

