import tensorflow.compat.v1 as tf
import numpy as np


from DQN import ActorDQN
from ActorParam import ActorDPG
from OrnsteinNoise import OUActionNoise
from Memory import ReplayBuffer
tf.disable_eager_execution()


class Agent(object):
    # beta must be "asymptotically negligible relative to alpha
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_continuous_actions=1, n_discrete_actions=5,
                 max_size=10000, layer1_size=64, layer2_size=32, layer3_size=32,
                 batch_size=64):
        self.n_discrete_actions = n_discrete_actions
        self.discrete_action_space = [i for i in range(n_discrete_actions)]
        self.gamma = gamma
        self.tau = tau  # for the soft update
        self.memory = ReplayBuffer(max_size)
        self.batch_size = batch_size
        self.sess = tf.Session()
        self.actor_DPG = ActorDPG(alpha, n_continuous_actions, 'ActorDPG', input_dims, self.sess,
                                  layer1_size, layer2_size, env.continuous_action_space.high) # TODO make continuous actor between 1 and -1
        self.actor_DQN = ActorDQN(beta, n_discrete_actions, n_continuous_actions, 'ActorDQN', input_dims, self.sess,
                                  layer1_size, layer2_size, layer3_size)

        self.target_actorDPG = ActorDPG(alpha, n_continuous_actions, 'TargetActorDPG', input_dims, self.sess,
                                        layer1_size, layer2_size, env.continuous_action_space.high)
        self.target_actorDQN = ActorDQN(beta, n_discrete_actions, n_continuous_actions, 'TargetActorDQN', input_dims,
                                        self.sess, layer1_size, layer2_size, layer3_size)

        self.noise = OUActionNoise(mu=np.zeros(n_continuous_actions))

        # define ops here in __init__ otherwise time to execute the op
        # increases with each execution.
        self.update_actorDQN = \
        [self.target_actorDQN.params[i].assign(
                      tf.multiply(self.actor_DQN.params[i], self.tau) \
                    + tf.multiply(self.target_actorDQN.params[i], 1. - self.tau))
         for i in range(len(self.target_actorDQN.params))]

        self.update_actorDPG = \
        [self.target_actorDPG.params[i].assign(
                      tf.multiply(self.actor_DPG.params[i], self.tau) \
                    + tf.multiply(self.target_actorDPG.params[i], 1. - self.tau))
         for i in range(len(self.target_actorDPG.params))]

        self.sess.run(tf.global_variables_initializer())

        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            old_tau = self.tau
            self.tau = 1.0
            self.target_actorDQN.sess.run(self.update_actorDQN)
            self.target_actorDPG.sess.run(self.update_actorDPG)
            self.tau = old_tau
        else:
            self.target_actorDQN.sess.run(self.update_actorDQN)
            self.target_actorDPG.sess.run(self.update_actorDPG)

    def remember(self, state, action_continuous, action_discrete, reward, new_state, done):
        done = 1 - done
        self.memory.add((state, action_continuous, action_discrete, reward, new_state, done))

    def choose_action(self, state, current_step, stop_step):
        state = state[np.newaxis, :]
        # get continuous action
        mu = self.actor_DPG.predict(state)  # returns list of list so get 0 ith element later on
        noise = self.noise()
        mu_prime = mu + noise
        action_continuous = mu_prime  #[0]
        assert action_continuous.shape == (1, 1)  # need this shape to run through actor_DQN
        # TODO this will need to be generalised by adding self.n_continuous_actions

        # get discrete action
        predict_discrete = self.actor_DQN.predict(state, action_continuous)
        action_discrete = self.eps_greedy_action(predict_discrete, current_step, stop_step)

        action_continuous = action_continuous[0]  # take it back to the correct shape
        return action_continuous, action_discrete

    def eps_greedy_action(self, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0.1):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        if random > explore_threshold:
            # paper uses uniform distribution
            # discrete_distribution = np.softmax(predict_discrete)
            action_discrete = np.random.choice(self.n_discrete_actions) #, p=discrete_distribution)
        else:
            action_discrete = np.argmax(predict_discrete)

        return action_discrete

    def best_action(self, state):
        state = state[np.newaxis, :]
        action_continuous = self.actor_DPG.predict(state)
        predict_discrete = self.actor_DQN.predict(state, action_continuous)
        action_discrete = np.argmax(predict_discrete)
        return action_continuous, action_discrete

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
        #action_values = np.array([self.discrete_action_space], dtype=np.int8)
        #discrete_action_indices = np.dot(action_discrete, action_values)

        Qvalues = self.target_actorDQN.predict(state, self.target_actorDPG.predict(state))
        Qvalues_next = self.target_actorDQN.predict(new_state,
                                                    self.target_actorDPG.predict(new_state))

        Q_target = Qvalues.copy()  # just to get shape and make difference 0 for everything that wasn't the taken action
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        Q_target[batch_index, action_discrete] = reward + self.gamma*np.max(Qvalues_next, axis=1)*done
        # done is 1 - done flag from environment
        assert Q_target.shape == (self.batch_size, self.n_discrete_actions), "Q target wrong shape"

        #action_discrete_matrix = np.zeros((self.batch_size, self.n_discrete_actions))
        #action_discrete_matrix[np.arange(self.batch_size), action_discrete] = 1

        _ = self.actor_DQN.train(state, action_continuous, Q_target)

        a_outs = self.actor_DPG.predict(state)
        grads = self.actor_DQN.get_action_gradients(state, a_outs)

        self.actor_DPG.train(state, grads[0])

        self.update_network_parameters()  # update target networks

    def save_models(self):
        self.actor_DPG.save_checkpoint()
        self.target_actorDPG.save_checkpoint()
        self.actor_DQN.save_checkpoint()
        self.target_actorDQN.save_checkpoint()

    def load_models(self):
        self.actor_DPG.load_checkpoint()
        self.target_actorDPG.load_checkpoint()
        self.actor_DQN.save_checkpoint()
        self.target_actorDQN.save_checkpoint()