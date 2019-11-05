
from tensorflow.keras.layers import Dense, Input, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adagrad, SGD, Adadelta, Nadam


class DQN_Agent:
    def __init__(self, lr, n_discrete_actions, n_continuous_actions, state_shape, name,
                 layer1_size=64, layer2_size=32, layer3_size=32, n_layers=4):
        self.lr = lr
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.name = name
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.layer3_size = layer3_size
        self.state_shape = state_shape
        self.n_layers = n_layers
        self.model = self.build_network()


    def build_network(self):
        input_state = Input(shape=self.state_shape, name="input_state")
        input_parameters = Input(shape=(self.n_continuous_actions,), name="input_parameters")
        flat_input_state = Flatten(name="flat_input_state")(input_state)
        flat_input_parameters = Flatten(name="flat_input_parameters")(input_parameters)
        inputs = Concatenate(name="concat")([flat_input_state, flat_input_parameters])

        dense1 = Dense(self.layer1_size, activation='relu', name="dense1")(inputs)
        dense2 = Dense(self.layer2_size, activation='relu', name="dense2")(dense1)
        dense3 = Dense(self.layer3_size, activation='relu', name="dense3")(dense2)
        if self.n_layers == 4:
            dense4 = Dense(self.layer3_size, activation='relu', name="dense4")(dense3)
            output = Dense(self.n_discrete_actions, activation='linear', name="output")(dense4)
        else:
            output = Dense(self.n_discrete_actions, activation='linear', name ="output")(dense3)

        model = Model(inputs=[input_state, input_parameters], outputs=output)
        optimizer = RMSprop(lr=self.lr)
        #optimizer = Adagrad(lr=self.lr) # super bad
        #optimizer = SGD(lr=self.lr, momentum=0.9, decay=0.01)
        #optimizer = Adadelta()  #shocking
        #optimizer = Nadam(lr=self.lr)

        return model, optimizer



