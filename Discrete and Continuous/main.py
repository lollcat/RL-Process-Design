
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
from tensorflow.keras.optimizers import RMSprop
set_floatx('float64')
import numpy as np
import multiprocessing
import concurrent.futures
import itertools
from Nets.P_actor import ParameterAgent

import time
from Utils.tester import Tester
from Utils.utils import Plotter, Visualiser
import matplotlib
import matplotlib.pyplot as plt
import re
"""
CONFIG
"""
allow_submit = False
reward_n = 0
product_all = True
decay = False
normal_cost = True

new_architecture = False
multiple_explore = True
freeze_point = True
freeze_train_factor = 1
sparse_reward = True
dueling_layer = True

if product_all is True:
    assert product_all != allow_submit

config = f"Config: \n fancy_arch:{new_architecture} \n freeze:{freeze_point} \n reward:{reward_n} \n " \
         f"submit:{allow_submit} \n decay:{decay} \n sparse:{sparse_reward} \n explore:{multiple_explore} " \
         f" \n constrain_product:{product_all} \ndueling:{dueling_layer} \n normal_cost:{normal_cost }"
config_string = re.sub("\n", "", config)
config_string = re.sub(" ", "", config_string)
config_string = re.sub(":", "_", config_string)

max_global_steps = 400
alpha = 0.0001
beta = alpha*10
steps_per_update = 5

"""Imports depending on config"""
if sparse_reward is True:
    from Env.Simulator_New import Simulator
else:
    from Env.Simulator_new_reward import Simulator

if product_all is False:
    from Workers.Worker_constrained import Worker
else:
    from Workers.Worker_constrained_6 import Worker
if new_architecture is True:
    from Nets.DQN_dueling_new_structure import DQN_Agent
elif dueling_layer is True and new_architecture:
    from Nets.DQN_dueling import DQN_Agent
else:
    from Nets.DQN import DQN_Agent


"""
OTHER INPUTS
"""
env = Simulator(allow_submit=allow_submit, metric=reward_n, normal_cost=normal_cost)
n_continuous_actions = env.continuous_action_space.shape[0]
n_discrete_actions = env.discrete_action_space.n
state_shape = env.observation_space.shape
layer1_size = 100
layer2_size = 50
layer3_size = 50
num_workers = multiprocessing.cpu_count()
global_counter = itertools.count()
returns_list = []

"""More stuff dependant on config"""
if freeze_point is True:
    global_counter2 = itertools.count()
    max_global_steps2 = max_global_steps*freeze_train_factor

if decay is True:
    param_decay = beta / max_global_steps
    dqn_decay = alpha / max_global_steps2
else:
    param_decay = False
    dqn_decay = False

"""GO GO GO"""
# Build Models
# Build Models
with tf.device('/CPU:0'):
    param_model, param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                    "Param_model", layer1_size=layer1_size,
                                                                    layer2_size=layer2_size).build_network()
    dqn_model, dqn_optimizer = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                               layer1_size, layer2_size, layer3_size).build_network()
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
            env=Simulator(allow_submit=allow_submit, metric=reward_n, normal_cost=normal_cost),
            max_global_steps=max_global_steps,
            returns_list=returns_list,
            multiple_explore=multiple_explore,
            n_steps=steps_per_update)
        workers.append(worker)


    coord = tf.train.Coordinator()
    worker_fn = lambda worker_: worker_.run(coord)
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        executor.map(worker_fn, workers, timeout=10)

    run_time = time.time() - start_time
    print(f'runtime part 1 is {run_time/60} min')

    if freeze_point is True:
        freeze_point = len(returns_list)
        """
        NOW DQN WITH PARAM NET FROZEN
        """
        # now keep param constant
        # Create Workers
        start_time = time.time()
        workers = []
        for worker_id in range(num_workers):
            worker = Worker(
                name=worker_id,
                global_network_P=param_model,
                global_network_dqn=dqn_model,
                global_optimizer_P=param_optimizer,
                global_optimizer_dqn=RMSprop(lr=alpha),
                global_counter=global_counter2,
                env=Simulator(allow_submit=allow_submit, metric=reward_n, normal_cost=normal_cost),
                max_global_steps=max_global_steps2,
                returns_list=returns_list,
                multiple_explore=multiple_explore,
                freeze=True,
                n_steps=steps_per_update)
            workers.append(worker)


        coord = tf.train.Coordinator()
        worker_fn = lambda worker_: worker_.run(coord)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            executor.map(worker_fn, workers, timeout=10)

        run_time2 = time.time() - start_time
        print(f'runtime part 2 is {run_time2/60} min')


for i in range(100):
    env = Tester(param_model, dqn_model, Simulator(allow_submit=allow_submit, metric=reward_n,
                                                   normal_cost=normal_cost), product_all=product_all).test()
    if reward_n is 0:
        returns_list.append(env.Performance_metric)
    else:
        returns_list.append(env.Performance_metric2)


plotter = Plotter(returns_list, len(returns_list) - 1, config_string, metric=reward_n,
                  freeze_point=freeze_point, show_heuristics=normal_cost)
if env.Performance_metric > plotter.by_lightness:
    matplotlib.rcParams['figure.dpi'] = 800
    plotter.plot(save=True)

    BFD = Visualiser(env).visualise()
    fig1, ax1 = plt.subplots()
    ax1.imshow(BFD)
    ax1.axis("off")
    fig1.savefig(f"Data_Plots/{config_string}BFD.png", bbox_inches='tight')


print(env.split_order)
print(env.sep_order)
print(env.Performance_metric)
print(env.Performance_metric2)
print(config)

"""
param_model.save("Nets/Agent_improved/param_model.h5")
dqn_model.save("Nets/Agent_improved/dqn_model.h5")
"""

"""
param_model.save("Nets/Agent_improved/param_model.h5")
dqn_model.save("Nets/Agent_improved/dqn_model.h5")
"""