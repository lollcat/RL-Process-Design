import tensorflow as tf
from OrnsteinNoise import OUActionNoise
from memory import ReplayBuffer
from DQN import DQN_Agent
from P_actor import ParameterAgent
from tensorflow.keras.models import clone_model


class PDQN_Agent:
    def __init__(self, alpha, beta, n_discrete_actions, n_continuous_actions, state_shape, tau, max_size=10000,
                 layer1_size=64, layer2_size=32, layer3_size=32):
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.state_shape = state_shape
        self.tau = tau
        self.memory = ReplayBuffer(max_size)

        self.dqn_model = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                                   layer1_size, layer2_size, layer3_size).build_network()
        self.param_model, self.param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                "Param_model", layer1_size=layer1_size,
                                                                layer2_size=layer2_size).build_network()

        self.target_dqn = clone_model(self.dqn_model)
        self.target_param = clone_model(self.param_model)

        self.update_actorDQN = \
            [self.target_dqn.trainable_weights[i].assign(
                tf.multiply(self.dqn_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_dqn.trainable_weights[i], 1-self.tau))
             for i in range(len(self.target_dqn.trainable_weights))]
        self.update_actorParam = \
            [self.target_param.trainable_weights[i].assign(
                tf.multiply(self.param_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_param.trainable_weights[i], 1-self.tau))
             for i in range(len(self.target_param.trainable_weights))]


test = PDQN_Agent(0.0001, 0.001, 5, 1, (5,), 100000)