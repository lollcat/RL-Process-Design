# https://www.udemy.com/course/deep-reinforcement-learning-in-python

from tensorflow.keras.backend import set_floatx
set_floatx('float64')
import numpy as np
from utils import Plotter
from DistillationSimulator import Simulator
import multiprocessing
import threading
import itertools
from P_actor import ParameterAgent
import tensorflow as tf

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
max_global_steps = 1000
steps_per_update = 5
num_workers = multiprocessing.cpu_count

global_counter = itertools.count
returns_list = []

# Build Models
param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                "Param_model", layer1_size=layer1_size,
                                                                layer2_size=layer2_size).build_network()
dqn_model = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                           layer1_size, layer2_size, layer3_size).build_network()

# Create Workers
workers = []
for worker_id in range(num_workers):
    worker = Worker(
        name=f'worker {worker_id}',
        param_model=param_model,
        dqn_model=dqn_model,
        param_optimizer=param_optimizer,
        dqn_optimizer=dqn_optimizer,
        global_counter=global_counter,
        env=Simulator(),
        max_global_steps=max_global_steps)
    workers.append(worker)

coord = tf.train.Coordinator()
# where to put @tf.function?
worker_threads = []
for worker in workers:
    worker_fn = lambda: worker.run(coord, steps_per_update)
    t = threading.Thread(target=worker_fn)
    t.start()
    worker_threads.append(t)

coord.join(worker_threads, stop_grace_period_secs=300)








