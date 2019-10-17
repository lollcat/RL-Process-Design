import numpy as np
import tensorflow as tf
from OrnsteinNoise import OUActionNoise
from tensorflow.keras.models import clone_model
import time
from memory import Memory

class Worker:
    def __init__(self, name, global_network_P, global_network_dqn, global_optimizer_P, global_optimizer_dqn,
                 global_counter, env, max_global_eps, score_history, batch_size=32, gamma=0.99, max_memory_len=500):
        self.name = name
        self.global_network_P = global_network_P
        self.global_network_dqn = global_network_dqn
        self.global_optimizer_P = global_optimizer_P
        self.global_optimizer_dqn = global_optimizer_dqn
        self.global_counter = global_counter
        self.env = env
        self.state = self.env.reset()
        self.max_global_eps = max_global_eps
        self.global_ep = 0
        self.score_history = score_history
        self.gamma = gamma
        self.noise = OUActionNoise(mu=np.zeros(env.continuous_action_space.shape[0]))
        self.n_discrete_actions = env.discrete_action_space.n
        self.start_time = time.time()
        self.batch_size = batch_size

        self.local_param_model = clone_model(global_network_P)
        self.local_param_model.set_weights(global_network_P.get_weights())
        self.local_dqn_model = clone_model(global_network_dqn)
        self.local_dqn_model.set_weights(global_network_dqn.get_weights())

        self.memory = Memory(max_memory_len)

    def run(self, coord):
        try:
            while not coord.should_stop():
                # Collect some experience
                self.run_episode()

                # Stop once the max number of global steps has been reached
                if self.max_global_eps is not None and self.global_ep >= self.max_global_eps:
                    coord.request_stop()
                    return f'worker {self.name}, step: {self.global_ep}'
        except tf.errors.CancelledError:
            return f'worker {self.name} tf.errors.CancelledError'

    def choose_action(self, state, current_step, stop_step):
        state = state[np.newaxis, :]
        # get continuous action
        mu = self.local_param_model.predict(state)  # returns list of list so get 0 ith element later on
        noise = self.noise()
        mu_prime = mu + noise
        action_continuous = min(mu_prime, np.array([[0.999]]))  # cannot have a split above 1
        assert action_continuous.shape == (1, 1)  # need this shape to run through actor_DQN
        # TODO this will need to be generalised by adding self.n_continuous_actions

        # get discrete action
        predict_discrete = self.local_dqn_model.predict([state, action_continuous])
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

    def run_episode(self):
        done = False
        score = 0
        while not done:
            action = self.choose_action(self.state, self.global_ep, self.max_global_eps/2)
            action_continuous, action_discrete = action
            next_state, reward, done, info = self.env.step(action)
            self.memory.add([self.state, action_continuous, action_discrete, reward, next_state, done])
            score += reward
            self.state = next_state

            if done:
                self.score_history.append(score)
                #print(f"Worker: {self.name} Score is {score}")
                self.state = self.env.reset()

        self.global_ep = next(self.global_counter)

        if self.max_global_eps / 40 % (self.global_ep + 1) == 0 and self.global_ep > 100:
            print(f'global counter: {self.global_ep}/{self.max_global_eps} \n')
            elapsed_time = time.time() - self.start_time
            remaining_time = elapsed_time * (self.max_global_eps - self.global_ep) / max(self.global_ep,
                                                                                             1)
            print(f'elapsed time: {elapsed_time / 60} min \n remaining time {remaining_time / 60} min \n')
            running_avg = np.mean(self.score_history[-100:])
            print(f'running average is {running_avg}')
        return

    def remember(self, state, action_continuous, action_discrete, reward, new_state, done):
        done = 1 - done
        self.memory.add((state, action_continuous, action_discrete, reward, new_state, done))

    #@tf.function
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

        with tf.GradientTape(persistent=True) as tape:
            predict_param = self.local_param_model(state)
            Qvalues = self.local_dqn_model([state, predict_param])
            loss_param = - tf.reduce_sum(Qvalues)

            Qvalues_next = self.local_dqn_model.predict([new_state, self.local_param_model.predict(new_state)])
            target_dqn = Qvalues.copy()  # get the shape
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            target_dqn[batch_index, action_discrete] = reward + self.gamma * np.max(Qvalues_next, axis=1) * done

            loss_dqn = tf.keras.losses.MSE(Qvalues, target_dqn)
        gradient_param = tape.gradient(loss_param, self.local_param_model.trainable_weights)
        gradient_dqn = tape.gradient(loss_dqn, self.local_dqn_model.trainable_weights)
        self.global_optimizer_P.apply_gradients(zip(gradient_param,
                                                    self.global_network_P.trainable_weights))
        self.global_optimizer_dqn.apply_gradients(zip(gradient_dqn,
                                                      self.global_network_dqn.trainable_weights))

        self.local_param_model.set_weights(self.global_network_P.get_weights())
        self.local_dqn_model.set_weights(self.global_network_dqn.get_weights())
