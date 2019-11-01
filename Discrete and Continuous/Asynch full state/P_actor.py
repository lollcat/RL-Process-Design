
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model



class ParameterAgent:
    def __init__(self, lr, n_continuous_actions, state_shape, name, layer1_size=64, layer2_size=32):
        self.lr = lr
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.state_shape = state_shape
        self.model, self.optimizer = self.build_network()

    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        flat_input_state = Flatten(name="flat_input_state")(input_state)
        dense1 = Dense(self.layer1_size, activation = 'relu', name="dense1")(flat_input_state)
        dense2 = Dense(self.layer2_size, activation = 'relu', name="dense2")(dense1)

        output = Dense(self.n_continuous_actions, activation='tanh', name="output")(dense2)

        model = Model(inputs=input_state, outputs=output)
        optimizer = RMSprop(lr=self.lr, decay=1e-6)

        return model, optimizer
