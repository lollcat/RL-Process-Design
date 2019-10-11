from DQN import DQN_Agent
from P_actor import ParameterAgent
import numpy as np
import tensorflow as ft


class Step:  # Stores a step
    def __init__(self, state, action_continuous, action_discrete, reward, next_state, done):
        self.state = state
        self.action_continuous = action_continuous
        self.action_discrete = action_discrete
        self.reward  = reward
        self.next_state = next_state
        self.done = done

#def copy_weights

def update_global_network(local_network, global_network):

    local_gradients, _ = zip(*local_network.gradients)