#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#tf.debugging.set_log_device_placement(True)
#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.keras.backend import set_floatx
set_floatx('float64')
import numpy as np
from utils import Plotter
from DistillationSimulator import Simulator
import multiprocessing
import threading
import itertools
from P_actor import ParameterAgent
from DQN import DQN_Agent
from Worker import Worker




"""
KEY INPUTS
"""
alpha = 0.0001
beta = 0.001
env = Simulator()
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape
layer1_size = 64
layer2_size = 32
layer3_size = 32
max_global_steps = 100
steps_per_update = 5
num_workers = multiprocessing.cpu_count()

global_counter = itertools.count()
returns_list = []

# Build Models
param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                "Param_model", layer1_size=layer1_size,
                                                                layer2_size=layer2_size).build_network()
dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                           layer1_size, layer2_size, layer3_size).build_network()

worker = Worker(
  name=f'worker {1}',
  global_network_P=param_model,
  global_network_dqn=dqn_model,
  global_optimizer_P=param_optimizer,
  global_optimizer_dqn=dqn_optimizer,
  global_counter=global_counter,
  env=Simulator(),
  max_global_steps=max_global_steps)

coord = tf.train.Coordinator()
worker.run(coord)