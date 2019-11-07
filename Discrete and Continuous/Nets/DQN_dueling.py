
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Nadam
import tensorflow as tf


class DQN_Agent:
    def __init__(self, lr, n_discrete_actions, n_continuous_actions, state_shape, name,
                 layer1_size=64, layer2_size=32, layer3_size=32, decay=False, optimizer_type=RMSprop):
        self.lr = lr
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.state_shape = state_shape
        self.optimizer_type = optimizer_type
        self.decay = decay
        self.model = self.build_network()



    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        input_parameters = Input(shape=(self.n_continuous_actions,), name="input_parameters")
        flat_input_state = Flatten(name="flat_input_state")(input_state)
        flat_input_parameters = Flatten(name="flat_input_parameters")(input_parameters)
        inputs = Concatenate(name="concat")([flat_input_state, flat_input_parameters])

        dense1 = Dense(self.layer1_size, activation='relu', name="dense1")(inputs)
        dense2 = Dense(self.layer2_size, activation= 'relu', name="dense2")(dense1)
        dense3 = Dense(self.layer3_size, activation='relu', name="dense3")(dense2)

        value_fc = Dense(self.layer3_size, activation='relu', name="value_fc")(dense3)
        value = Dense(1, activation='linear', name="value")(value_fc)

        advantage_fc = Dense(self.layer3_size, activation='relu', name='advantage_fc')(dense3)
        advantage = Dense(self.n_discrete_actions, activation='linear')(advantage_fc)

        output = value + tf.math.subtract(advantage, tf.math.reduce_mean(advantage, axis=1, keepdims=True))

        model = Model(inputs=[input_state, input_parameters], outputs=output)
        if self.decay is False:
            optimizer = self.optimizer_type(lr=self.lr)
        else:
            optimizer = self.optimizer_type(lr=self.lr, decay=self.decay)
        return model, optimizer



