
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Nadam
import tensorflow as tf


class DQN_Agent:
    def __init__(self, lr, n_discrete_actions, n_continuous_actions, state_shape, name ="DQN_model",
                 layer1_size="NA", layer2_size="NA", layer3_size="NA", decay=False, stream_len=6):
        self.lr = lr
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.state_shape = state_shape
        self.optimizer_type = RMSprop
        self.decay = decay
        self.stream_len = stream_len
        self.model = self.build_network()


    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        input_parameters = Input(shape=(self.n_continuous_actions,), name="input_parameters")
        flat_input_parameters = Flatten(name="flat_input_parameters")(input_parameters)

        slices = []
        fc1s = []
        fc2s = []
        for i in range(self.stream_len):
            slices.append((Lambda(lambda x: x[:, i, :])(input_state)))
            fc1s.append(Dense(self.stream_len, activation=None, name=f"fc1{i}")(slices[i]))
            fc2s.append(Dense(self.stream_len-1, activation='relu', name=f"fc2{i}")(fc1s[i]))
        input_fcs_output = Concatenate(name='input_fcs_output')(fc2s)

        inputs = Concatenate(name="concat")([input_fcs_output, flat_input_parameters])

        dense1 = Dense(self.n_discrete_actions, activation='relu', name="dense1")(inputs)
        dense2 = Dense(self.n_discrete_actions, activation= 'relu', name="dense2")(dense1)
        dense3 = Dense(self.n_discrete_actions, activation='relu', name="dense3")(dense2)

        value_fc = Dense(self.n_discrete_actions, activation='relu', name="value_fc")(dense3)
        value = Dense(1, activation='linear', name="value")(value_fc)

        advantage_fc = Dense(self.n_discrete_actions, activation='relu', name='advantage_fc')(dense3)
        advantage = Dense(self.n_discrete_actions, activation='linear')(advantage_fc)

        output = value + tf.math.subtract(advantage, tf.math.reduce_mean(advantage, axis=1, keepdims=True))

        model = Model(inputs=[input_state, input_parameters], outputs=output)
        if self.decay is False:
            optimizer = self.optimizer_type(lr=self.lr)
        else:
            optimizer = self.optimizer_type(lr=self.lr, decay=self.decay)
        return model, optimizer

"""
from Env.Simulator_New import Simulator
env = Simulator()
state_shape1 = env.observation_space.shape
classs = DQN_Agent(0.001, 30, 1, state_shape1)
dqn_model, dqn_optimizer = classs.build_network()
plot_model(dqn_model, "dqn_new_structure.png")
"""