
from Env.Simulator_New import Simulator

env = Simulator()
for i in range(env.split_option_n):
    selected_stream = env.n_outlet_streams - 1
    selected_compound = i
    discrete_action = selected_stream*env.split_option_n + selected_stream
    continuous_action = 1
    action = ()
    env.step([continuous_action, discrete_action])

performance_ordered = env.Performance_metric
sequence_1 = env.sep_order
print(f'Performance metric sep by lightness is {performance_ordered}')

env.reset()
flowrated_order = [2, 1, 4, 0, 3]
sep_order = [0, 0, 1, 0, 1]
for i in range(env.split_option_n):
    selected_stream = sep_order[i]
    selected_compound = flowrated_order[i]
    discrete_action = selected_stream*env.split_option_n + selected_stream
    continuous_action = 1
    action = ()
    env.step([continuous_action, discrete_action])
performance_flowrate = env.Performance_metric
sequence_2 = env.sep_order
print(f'Performance metric sep by flowrate is {performance_flowrate}')

env.reset()
flowrated_order = [2, 1, 4, 0, 3]
sep_order = [0, 0, 1, 0, 1]
for i in range(env.split_option_n):
    selected_stream = sep_order[i]
    selected_compound = flowrated_order[i]
    discrete_action = selected_stream*env.split_option_n + selected_stream
    continuous_action = 1
    action = ()
    env.step([continuous_action, discrete_action])
performance_flowrate = env.Performance_metric
sequence_2 = env.sep_order
print(f'Performance metric sep by flowrate is {performance_flowrate}')