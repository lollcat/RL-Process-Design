
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
from tensorflow.keras.models import load_model
set_floatx('float64')
import numpy as np
from Env.Simulator_New import Simulator
import multiprocessing
import concurrent.futures
import itertools
from Nets.P_actor import ParameterAgent
#from Nets.DQN import DQN_Agent
from Nets.DQN_dueling import DQN_Agent
from Workers.Worker_memory import Worker
import time
from Utils.tester import Tester
from Utils.utils import Plotter, Visualiser
import matplotlib
import matplotlib.pyplot as plt

"""
CONFIG
"""
#from Env.Simulator_New import Simulator
from Env.Simulator_new_reward import Simulator
allow_submit = True
reward_n = 1
decay = True

"""
KEY INPUTS
"""
alpha = 0.0001
beta = alpha*10
max_global_eps = 500000

env = Simulator(allow_submit=allow_submit, metric=reward_n)
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape
layer1_size = 100
layer2_size = 50
layer3_size = 50

if decay is True:
    param_decay = beta/max_global_eps
    dqn_decay = alpha/max_global_eps
else:
    param_decay = False
    dqn_decay = False

num_workers = multiprocessing.cpu_count()
global_counter = itertools.count()
returns_list = []

# Build Models
with tf.device('/CPU:0'):
    param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                  "Param_model", layer1_size=layer1_size,
                                                  layer2_size=layer2_size, decay=param_decay ).build_network()
    dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                                         layer1_size, layer2_size, layer3_size, decay=dqn_decay).build_network()
    #param_model = load_model("param_model.h5")
    #dqn_model = load_model("dqn_model.h5")
    # Create Workers
    start_time = time.time()
    workers = []
    for worker_id in range(num_workers):
        worker = Worker(
            name=worker_id,
            global_network_P=param_model,
            global_network_dqn=dqn_model,
            global_optimizer_P=param_optimizer,
            global_optimizer_dqn=dqn_optimizer,
            global_counter=global_counter,
            env=Simulator(allow_submit=allow_submit, metric=reward_n),
            max_global_steps=max_global_eps,
            returns_list=returns_list,
            batch_size=15,
            max_len=150)
        workers.append(worker)


    coord = tf.train.Coordinator()
    worker_fn = lambda worker_: worker_.run(coord)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(worker_fn, workers, timeout=10)

run_time = time.time() - start_time
print(f'runtime is {run_time/60} min')

for i in range(100):
    env = Tester(param_model, dqn_model, Simulator(allow_submit=allow_submit, metric=reward_n)).test()
    returns_list.append(env.Performance_metric2)
print(env.split_order)
print(env.sep_order)
print(env.Performance_metric)
print(env.Performance_metric2)

plotter = Plotter(returns_list, len(returns_list) - 1, metric=reward_n)
plotter.plot()
"""
if env.Performance_metric > plotter.by_lightness:
    BFD = Visualiser(env).visualise()
    matplotlib.rcParams['figure.dpi'] = 800
    fig, ax = plt.subplots()
    ax.imshow(BFD)
    ax.axis("off")
    #fig.savefig(f"Data_Plots/With_freeze/reward{reward_n}/freeze_reward{reward_n}allowsubmit{allow_submit}BFD.png", bbox_inches='tight')
print(env.split_order)
print(env.sep_order)
print(env.Performance_metric)
print(env.Performance_metric2)
print(f"using reward {reward_n}")
print(f"submit allow {allow_submit}")
"""
