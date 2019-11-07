import numpy as np
from gym import spaces
from scipy.optimize import fsolve
from Env.Sizing.column_sizing import columnsize
from Env.Sizing.dc_annual_cost import DCoperationcost
from Env.Sizing.dc_capital_cost import expensiveDC

"""  Natural Gas Problem
        self.compound_names = ["Methane", "Ethane", "Propane", "Isobutane", "Butane",  "Pentane+"]
        Molar_weights = np.array([16.043, 30.07, 44.097, 58.124, 58.124, 72.151])
        Heating_value = np.array([55.6,  51.9, 50.4, 49.5, 49.4,   55.2]) #MJ/kg
        Price_per_MBtu = np.array([2.83, 2.54, 4.27, 5.79, 5.31,  10.41])  #  $/Million Btu  # TODO update for methane
        
                # Anotoine data from YAWS
        self.Antoine_a1 = np.array([6.84566, 6.95335, 7.01887, 6.93388, 7.00961,  7.00877])
        self.Antoine_a2 = np.array([435.621, 699.106, 889.864, 953.92, 1022.48,  1134.15 ])
        self.Antoine_a3 = np.array([271.361, 260.264, 257.084, 247.077, 248.145,  238.678])
        self.initial_state = np.array([0.75, 0.15, 0.1, 0.01, 0.02,  0.03])*5269 # Flowrates kmol/hr
        self.product_prices = Molar_weights*Heating_value*1000/1.055*Price_per_MBtu/1000000*exchange_rate
"""

class Simulator:
    def __init__(self, metric=0, allow_submit=False, normal_cost=True):
        self.allow_submit = allow_submit
        self.metric = metric
        # Compound Data
        self.compound_names = ["Ethane", "Propylene", "Propane", "1-butene", "n-butane", "n-pentane"]
        self.feed = np.array([3, 5,  12, 2, 9, 7])
        self.max_outlet_streams = 6
        self.initial_state = np.zeros((self.max_outlet_streams, self.feed.shape[0]))
        self.initial_state[0] = self.feed
        self.state = self.initial_state.copy()
        self.n_outlet_streams = 1
        self.split_option_n = self.feed.shape[0] - 1
        self.stream_table = [self.feed]

        # to keep track of which stream numbers correspond to outlet steams
        self.state_streams = np.zeros(self.max_outlet_streams) - 1
        self.state_streams[0] = 0

        self.discrete_action_size = self.max_outlet_streams * self.split_option_n  # action selects stream & LK
        if allow_submit is True:
            self.discrete_action_size += 1
        continuous_action_number = 1

        # spaces for mixed action space?
        self.discrete_action_space = spaces.Discrete(self.discrete_action_size)
        self.continuous_action_space_input = spaces.Box(low=-1, high=1, shape=(continuous_action_number,))
        self.continuous_action_space = spaces.Box(low=0.8, high=0.999, shape=(continuous_action_number,))
        self.observation_space = spaces.Box(low=0, high=9.1, shape=self.initial_state.shape)


        density = np.array([0.95, 170, 1.4, 0.9619*56.108, 1.9, 20])  # kg/m3
        Molar_weights = np.array([30.07,  42.08,  44.097,  56.108,  58.124,  72.15])
        self.molar_density = density * Molar_weights
        Heating_value = np.array([51.9,   49.0,   50.4,    48.5,    49.4,    48.6])  # MJ/kg
        if normal_cost is True:
            Price_per_MBtu = np.array([2.54,  17.58,  4.27,    29.47,   5.31,    13.86])  # $/Million Btu
        else:
            Price_per_MBtu = np.array([2.54, 17.58, 0, 0, 5.31, 13.86])

        self.Antoine_a1 = np.array([6.95335, 7.01612, 7.01887, 7.0342,  7.00961, 7.00877])
        self.Antoine_a2 = np.array([699.106, 860.992, 889.864, 1013.6,  1022.48, 1134.15])
        self.Antoine_a3 = np.array([260.264, 255.895, 257.084, 250.292, 248.145, 238.678])

        exchange_rate = 15.23  #  $/R
        self.product_prices = Molar_weights*Heating_value*1000/1.055*Price_per_MBtu/1000000*exchange_rate


        self.product_streams =[]
        self.column_streams = []
        self.sep_order = []
        self.split_order = []
        self.column_conditions = []  # pressure, tops temperature, bots temperature
        self.column_dimensions = []  # Nstages, Reflux ratio
        self.capital_cost = []
        self.revenue = 0
        self.revenue_by_stream = []
        self.Profit = 0
        self.Performance_metric = 0
        self.Performance_metric2 = 0

    def step(self, action):
        info = []
        reward = 0

        # get actions
        action_continuous, action_discrete = action
        if action_discrete == self.discrete_action_size - 1 and self.allow_submit is True:
            done = True
            print("end early")
            self.classify_streams()
            self.revenue = self.revenue_calculator(self.product_streams)
            self.Performance_metric = 10 - sum(self.capital_cost) / max(self.revenue, 1) # prevent 0 revenue from effecting things
            self.Performance_metric2 = (self.revenue - sum(self.capital_cost)/10)/1e7

            if self.metric == 0:
                reward = max(self.Performance_metric, -10)
            elif self.metric == 1:
                reward = max(self.Performance_metric2, -10)

            return self.state, reward, done, info

        Light_Key = action_discrete % self.split_option_n
        Selected_stream = int(action_discrete/self.split_option_n)
        LK_split = self.action_continuous_definer(action_continuous)
        self.split_order.append(LK_split)
        self.sep_order.append(Light_Key)
        selected_stream_number = self.state_streams[Selected_stream]

        done = False
        feed = self.state[Selected_stream].copy()



        Heavy_Key = Light_Key + 1
        HK_split = 1 - LK_split
        # HK_split = action[2]
        tops = np.zeros(feed.shape)
        tops[:Light_Key + 1] = feed[:Light_Key + 1]
        tops[Light_Key] = tops[Light_Key] * LK_split
        tops[Heavy_Key] = feed[Heavy_Key] * HK_split
        bots = feed - tops

        LK_D = tops[Light_Key] / tops.sum()
        HK_D = tops[Heavy_Key] / tops.sum()
        LK_B = bots[Light_Key] / bots.sum()
        HK_B = bots[Heavy_Key] / bots.sum()
        assert LK_D < 1 and LK_D > 0
        assert LK_B < 1 and LK_B > 0
        assert HK_D < 1 and HK_D > 0
        assert HK_B < 1 and HK_B > 0

        self.stream_table.append(tops)
        self.stream_table.append(bots)
        self.column_streams.append((selected_stream_number, len(self.stream_table)-2, len(self.stream_table)-1))

        # fill state with new outlet streams, deleting the stream being seperated
        self.state[Selected_stream] = tops
        self.state_streams[Selected_stream] = len(self.stream_table)-2
        self.state[self.n_outlet_streams] = bots
        self.state_streams[self.n_outlet_streams] = len(self.stream_table)-1

        self.n_outlet_streams += 1

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
        liquidfeed = feed  # always completely condensed? # TODO check this
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

        volumetric_flow = np.sum(tops/self.molar_density)
        Length, Diameter = columnsize(n_stages_actual, volumetric_flow)
        capital_cost = expensiveDC(Diameter, Length, system_pressure, bots_temperature, n_stages_actual)

        if Length > 170 or Diameter > 17:  # limiting thresholds (would require custom column)
            capital_cost *= 10
        self.capital_cost.append(capital_cost)


        if self.n_outlet_streams >= self.max_outlet_streams:  # episode ends after max number of streams produced
            done = True

        if done is True:
            self.classify_streams()
            self.revenue = self.revenue_calculator(self.product_streams)
            self.Performance_metric = 10 - sum(self.capital_cost) / max(self.revenue, 0.01) # prevent 0 revenue from effecting things
            self.Performance_metric2 = (self.revenue - sum(self.capital_cost)/10)/1e7

            if self.metric == 0:
                reward = max(self.Performance_metric, -10)
            elif self.metric == 1:
                reward = max(self.Performance_metric2, -10)

        return self.state, reward, done, info

    def classify_streams(self):
        for i in range(self.n_outlet_streams):
            stream = self.state[i]
            purity = max(np.divide(stream, stream.sum()))
            recovery = max(np.divide(stream, self.feed))
            stream_is_product = purity > 0.96  # other conditions to add?
            if stream_is_product:
                self.product_streams.append(stream)

    def reset(self):
        self.state = self.initial_state.copy()
        self.stream_table = [self.initial_state.copy()]
        self.sep_order = []
        self.split_order = []
        self.column_conditions = []
        self.column_dimensions = []
        self.product_streams = []
        self.revenue = 0
        self.revenue_by_stream = []
        self.Profit = 0
        self.Performance_metric = 0
        self.Performance_metric2 = 0
        self.capital_cost = []
        self.column_streams = []

        self.state_streams = np.zeros(self.max_outlet_streams) - 1
        self.state_streams[0] = 0
        self.n_outlet_streams = 1

        return self.state

    def render(self, mode='human'):
        print(f'stream {self.state} is seperated with an LK of {self.sep_order[-1]} '
              f'with a split of {self.split_order[-1]}')

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
        vapour_pressures = (10**(self.Antoine_a1 - (np.divide(self.Antoine_a2,
                                                              (self.Antoine_a3 + temperature)))))*0.1333
        return vapour_pressures

    def Kvalues_calculator(self, temperature, system_pressure):
        vapour_pressure = self.vapour_pressure_calculator(temperature)
        Kvalues = vapour_pressure / system_pressure
        return Kvalues

    def revenue_calculator(self, product_streams):
        revenue = 0
        for stream in product_streams:
            stream_revenue = stream.max() * self.product_prices[np.argmax(stream)]*8000
            self.revenue_by_stream.append(stream_revenue)
            revenue += stream_revenue
        return revenue

    def run_random(self):
            action_continuous = self.continuous_action_space_input.sample()
            illegal_actions = self.illegal_actions(self.state)
            action_discrete = np.random.choice([i for i in range(self.discrete_action_size) if illegal_actions[i] == False])
            state, reward, done, _ = self.step([action_continuous, action_discrete])
            return state, reward, done, _

    def illegal_actions(self, state):
        state = state[np.newaxis, :]
        LK_legal1 = state[:, :, 0:-1] == 0
        LK_legal1 = LK_legal1.flatten(order="C")
        LK_legal2 = state[:, :, 1:] == 0
        LK_legal2 = LK_legal2.flatten(order="C")
        LK_legal = LK_legal1 + LK_legal2
        if self.allow_submit is True:
            if self.n_outlet_streams > 1:
                LK_legal = np.append(LK_legal, False)
            else:
                LK_legal = np.append(LK_legal, True)
        return LK_legal

"""
env = Simulator()
for i in range(5):
    print(env.run_random())
"""
