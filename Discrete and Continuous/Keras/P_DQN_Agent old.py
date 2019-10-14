import tensorflow as tf
from tensorflow.keras.models import clone_model, load_model
from tensorflow import reduce_sum
import numpy as np

from OrnsteinNoise import OUActionNoise
from memory import ReplayBuffer
from DQN import DQN_Agent
from P_actor import ParameterAgent

class Agent:
    def __init__(self, alpha, beta, n_discrete_actions, n_continuous_actions, state_shape, tau=0.001, batch_size=32,
                 gamma=0.99, max_size=10000, layer1_size=64, layer2_size=32, layer3_size=32):
        self.batch_size = batch_size
        self.n_discrete_actions = n_discrete_actions
        self.n_continuous_actions = n_continuous_actions
        self.state_shape = state_shape
        self.tau = tau
        self.gamma = gamma
        self.memory = ReplayBuffer(max_size)

        self.dqn_model = DQN_Agent(alpha, n_discrete_actions, n_continuous_actions, state_shape, "DQN_model",
                                   layer1_size, layer2_size, layer3_size).build_network()
        self.param_model, self.param_optimizer = ParameterAgent(beta, n_continuous_actions, state_shape,
                                                                "Param_model", layer1_size=layer1_size,
                                                                layer2_size=layer2_size).build_network()

        self.target_dqn = clone_model(self.dqn_model)
        self.target_param = clone_model(self.param_model)

        self.noise = OUActionNoise(mu=np.zeros(n_continuous_actions))

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0

            [self.target_dqn.trainable_weights[i].assign(
                tf.multiply(self.dqn_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_dqn.trainable_weights[i], 1 - self.tau))
                for i in range(len(self.target_dqn.trainable_weights))]

            [self.target_param.trainable_weights[i].assign(
                tf.multiply(self.param_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_param.trainable_weights[i], 1 - self.tau))
                for i in range(len(self.target_param.trainable_weights))]

            self.tau = old_tau
        else:
            [self.target_dqn.trainable_weights[i].assign(
                tf.multiply(self.dqn_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_dqn.trainable_weights[i], 1 - self.tau))
                for i in range(len(self.target_dqn.trainable_weights))]

            [self.target_param.trainable_weights[i].assign(
                tf.multiply(self.param_model.trainable_weights[i], self.tau)
                + tf.multiply(self.target_param.trainable_weights[i], 1 - self.tau))
                for i in range(len(self.target_param.trainable_weights))]

    def remember(self, state, action_continuous, action_discrete, reward, new_state, done):
        done = 1 - done
        self.memory.add((state, action_continuous, action_discrete, reward, new_state, done))

    def choose_action(self, state, current_step, stop_step):
        state = state[np.newaxis, :]
        # get continuous action
        mu = self.param_model.predict(state)  # returns list of list so get 0 ith element later on
        noise = self.noise()
        mu_prime = mu + noise
        action_continuous = min(mu_prime, np.array([[0.999]]))  # cannot have a split above 1
        assert action_continuous.shape == (1, 1)  # need this shape to run through actor_DQN
        # TODO this will need to be generalised by adding self.n_continuous_actions

        # get discrete action
        predict_discrete = self.dqn_model.predict([state, action_continuous])
        action_discrete = self.eps_greedy_action(predict_discrete, current_step, stop_step)

        action_continuous = action_continuous[0]  # take it back to the correct shape
        return action_continuous, action_discrete

    def eps_greedy_action(self, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0.1):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        if random < explore_threshold:
            # paper uses uniform distribution
            # discrete_distribution = np.softmax(predict_discrete)
            action_discrete = np.random.choice(self.n_discrete_actions)  # , p=discrete_distribution)
        else:
            action_discrete = np.argmax(predict_discrete)

        return action_discrete

    def best_action(self, state):
        state = state[np.newaxis, :]
        action_continuous = self.param_model.predict(state)
        predict_discrete = self.dqn_model.predict([state, action_continuous])
        action_discrete = np.argmax(predict_discrete)
        return action_continuous, action_discrete


    def train_param(self, state):
        with tf.GradientTape() as tape:
            predict_param = self.param_model(state)
            Qvalues = self.target_dqn([state, predict_param])
            loss = - reduce_sum(Qvalues, axis=1, keepdims=True)
            # get gradients of loss with respect to the param_model weights
        gradients = tape.gradient(loss, self.param_model.trainable_weights)
        self.param_optimizer.apply_gradients(zip(gradients, self.param_model.trainable_weights))
        return loss

    @tf.function
    def learn(self):
        if len(self.memory.buffer) < self.batch_size:  # first fill memory
            return
        batch = self.memory.sample(self.batch_size)
        state = np.array([each[0] for each in batch])
        action_continuous = np.array([each[1] for each in batch])
        action_discrete = np.array([each[2] for each in batch])
        reward = np.array([each[3] for each in batch])
        new_state = np.array([each[4] for each in batch])
        done = np.array([each[5] for each in batch])

        action_discrete = np.array(action_discrete)

        # train the continuous variable network on the Q values
        # also the Q values are used again later for training of DQN
        # see tf.function train_param
        loss_param = self.train_param(state)
        Qvalues = self.target_dqn.predict([state, self.target_param.predict(state)])
        Qvalues_next = self.target_dqn.predict([new_state, self.target_param.predict(new_state)])

        Q_target = Qvalues.copy()  # just to get shape and make difference 0 for everything that wasn't the taken action
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        Q_target[batch_index, action_discrete] = reward + self.gamma*np.max(Qvalues_next, axis=1)*done
        # done is 1 - done flag from environment
        assert Q_target.shape == (self.batch_size, self.n_discrete_actions), "Q target wrong shape"

        # now train dqn model
        _ = self.dqn_model.train_on_batch([state, action_continuous], Q_target)

        self.update_network_parameters()

    def save_models(self):
        self.param_model.save("param_model.h5")
        self.dqn_model.save("dqn_model.h5")
        self.target_param.save("target_param.h5")
        self.target_dqn.save("target_dqnv.h5")

    def load_models(self):
        self.param_model = load_model("param_model.h5")
        self.dqn_model = load_model("dqn_model.h5")
        self.target_param = load_model("target_param.h5")
        self.target_dqn = load_model("target_dqnv.h5")
