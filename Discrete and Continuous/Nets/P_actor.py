
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Nadam
from tensorflow.keras.utils import plot_model



class ParameterAgent:
    def __init__(self, lr, n_continuous_actions, state_shape, name, layer1_size=64, layer2_size=32,
                 decay=False, optimizer_type=RMSprop):
        self.lr = lr
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.state_shape = state_shape
        self.optimizer_type = optimizer_type
        self.decay = decay
        self.model, self.optimizer = self.build_network()


    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        flat_input_state = Flatten(name="flat_input_state")(input_state)
        dense1 = Dense(self.layer1_size, activation = 'relu', name="dense1")(flat_input_state)
        dense2 = Dense(self.layer2_size, activation = 'relu', name="dense2")(dense1)
        output = Dense(self.n_continuous_actions, activation='tanh', name="output")(dense2)

        model = Model(inputs=input_state, outputs=output)

        if self.decay is False:
            optimizer = self.optimizer_type(lr=self.lr)
        else:
            optimizer = self.optimizer_type(lr=self.lr, decay=self.decay)
        return model, optimizer

"""
        optimizer = Adagrad(lr=self.lr) #did superbad
        optimizer = SGD(lr=self.lr, momentum=0.9, decay=0.01) # medium - try with more data?
        optimizer = Adadelta() # shocking
        optimizer = Nadam(lr=self.lr)
"""
