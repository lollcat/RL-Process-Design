from OrnsteinNoise import OUActionNoise
from Memory import ReplayBuffer


class PDQN_Agent:
    def __init__(self, alpha, beta, n_discrete_actions, n_continuous_actions, state_shape, name, layer1_size=64, layer2_size=32,
                 layer3_size=32):
        self.lr = lr
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.state_shape = state_shape

