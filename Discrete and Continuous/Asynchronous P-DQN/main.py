# https://www.udemy.com/course/deep-reinforcement-learning-in-python

import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


from tensorflow.keras.backend import set_floatx
set_floatx('float64')
import numpy as np
from utils import Plotter
from DistillationSimulator import Simulator
import multiprocessing
#import threading
import concurrent.futures
import itertools
from P_actor import ParameterAgent
from DQN import DQN_Agent
from Worker import Worker
import time


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
max_global_steps = 20000
steps_per_update = 5
num_workers = multiprocessing.cpu_count()

global_counter = itertools.count()
returns_list = []

# Build Models
with tf.device('/CPU:0'):
    param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                    "Param_model", layer1_size=layer1_size,
                                                                    layer2_size=layer2_size).build_network()
    dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                               layer1_size, layer2_size, layer3_size).build_network()

    # Create Workers
    start_time = time.time()
    workers = []
    for worker_id in range(num_workers):
        worker = Worker(
            name=f'worker {worker_id}',
            global_network_P=param_model,
            global_network_dqn=dqn_model,
            global_optimizer_P=param_optimizer,
            global_optimizer_dqn=dqn_optimizer,
            global_counter=global_counter,
            env=Simulator(),
            max_global_steps=max_global_steps,
            returns_list=returns_list,
            n_steps=steps_per_update)
        workers.append(worker)


coord = tf.train.Coordinator()
worker_fn = lambda worker_: worker_.run(coord)
with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
    executor.map(worker_fn, workers, timeout=10)

run_time = time.time() - start_time
print(f'runtime is {run_time/60} min')

plotter = Plotter(returns_list, len(returns_list)-1)
plotter.plot()