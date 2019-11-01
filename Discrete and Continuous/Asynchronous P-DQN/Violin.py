
from Env.Simulator_New import Simulator
import matplotlib.pyplot as plt
from Worker_constrained import Worker
from tester import Tester
from P_actor import ParameterAgent
from DQN import DQN_Agent

with tf.device('/CPU:0'):
    param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                    "Param_model", layer1_size=layer1_size,
                                                                    layer2_size=layer2_size).build_network()
    dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                               layer1_size, layer2_size, layer3_size).build_network()

env = Tester(param_model, dqn_model, Simulator()).test()
