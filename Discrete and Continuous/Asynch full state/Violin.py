import tensorflow as tf
from Env.Simulator_New import Simulator
from tester import Tester
from P_actor import ParameterAgent
from DQN import DQN_Agent
from tensorflow.keras.utils import plot_model
from utils import Visualiser
import matplotlib.pyplot as plt
import matplotlib


alpha = 0.0001
beta = 0.001
env = Simulator()
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape
layer1_size = 100
layer2_size = 50
layer3_size = 50


with tf.device('/CPU:0'):
    param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                    "Param_model", layer1_size=layer1_size,
                                                                    layer2_size=layer2_size).build_network()
    dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                               layer1_size, layer2_size, layer3_size).build_network()

#plot_model(dqn_model, to_file='DQNmodel.png', show_shapes=True)
tester = Tester(param_model, dqn_model, Simulator())
#env_list = []
for _ in range(100):
    env = (tester.test())
    tester.env.reset()
"""
BFD = Visualiser(env).visualise()
matplotlib.rcParams['figure.dpi']= 800
fig, ax = plt.subplots()
ax.imshow(BFD)
fig.savefig("BFD")
"""