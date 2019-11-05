import numpy as np
import tensorflow as tf
from Utils.OrnsteinNoise import OUActionNoise
from Utils.memory import Memory
from tensorflow.keras.models import clone_model
import time
from scipy.special import softmax
import sys
import linecache

class Step:  # Stores a step
    def __init__(self, state, action_continuous, action_discrete, reward, next_state, done):
        self.state = state
        self.action_continuous = action_continuous
        self.action_discrete = action_discrete
        self.reward = reward
        self.next_state = next_state
        self.done = done


class Worker:
    def __init__(self, name, global_network_P, global_network_dqn, global_optimizer_P, global_optimizer_dqn,
                 global_counter, env, max_global_steps, returns_list, gamma=0.99, batch_size=10, max_len=200):
        self.name = name
        self.global_network_P = global_network_P
        self.global_network_dqn = global_network_dqn
        self.global_optimizer_P = global_optimizer_P
        self.global_optimizer_dqn = global_optimizer_dqn
        self.global_counter = global_counter
        self.env = env
        self.allow_submit = env.allow_submit
        self.state = self.env.reset()
        self.max_global_steps = max_global_steps
        self.global_step = 0
        self.returns_list = returns_list
        self.gamma = gamma
        self.noise = OUActionNoise(mu=np.zeros(env.continuous_action_space.shape[0]))
        self.n_discrete_actions = env.discrete_action_space.n
        self.start_time = time.time()

        self.local_param_model = clone_model(global_network_P)
        self.local_param_model.set_weights(global_network_P.get_weights())
        self.local_dqn_model = clone_model(global_network_dqn)
        self.local_dqn_model.set_weights(global_network_dqn.get_weights())

        self.memory = Memory(max_len)
        self.batch_size=batch_size

    def run(self, coord):
        try:
            while not coord.should_stop():
                # Collect some experience
                self.run_ep()
                # Update the global networks using local gradients
                self.update_global_parameters()

                # Stop once the max number of global steps has been reached
                if self.max_global_steps is not None and self.global_step >= self.max_global_steps:
                    coord.request_stop()
                    return f'worker {self.name}, step: {self.global_step}'

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

        # get discrete action
        predict_discrete = self.local_dqn_model.predict([state, action_continuous])
        action_discrete = self.eps_greedy_action(state, predict_discrete, current_step, stop_step)
        """
        if self.name % 2 == 0:
            action_discrete = self.eps_greedy_action(state, predict_discrete, current_step, stop_step)
        else:
            action_discrete = self.proportion_action(state, predict_discrete, current_step, stop_step)
        """

        action_continuous = action_continuous[0]  # take it back to the correct shape
        return action_continuous, action_discrete

    def eps_greedy_action(self, state, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0.05):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        illegal_actions = self.illegal_actions(state)
        predict_discrete[:, illegal_actions] = predict_discrete.min() - 1
        if random < explore_threshold:
            action_discrete = np.random.choice(
                   [i for i in range(self.n_discrete_actions) if illegal_actions[i] == False])
        else:
            action_discrete = np.argmax(predict_discrete)

        return action_discrete

    def proportion_action(self, state, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0.05):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        illegal_actions = self.illegal_actions(state)
        predict_discrete[:, illegal_actions] = predict_discrete.min() - 1
        if random < explore_threshold:
            discrete_distribution = softmax(predict_discrete)[0]
            discrete_distribution[illegal_actions] = 0
            action_discrete = np.random.choice(self.n_discrete_actions,
                                               p=discrete_distribution / discrete_distribution.sum())
        else:
            action_discrete = np.argmax(predict_discrete)
        return action_discrete

    def illegal_actions(self, state):
        LK_legal1 = state[:, :, 0:-1] == 0
        LK_legal1 = LK_legal1.flatten(order="C")
        LK_legal2 = state[:, :, 1:] == 0
        LK_legal2 = LK_legal2.flatten(order="C")
        LK_legal = LK_legal1 + LK_legal2
        if self.allow_submit is True:
            if self.env.n_outlet_streams > 1:
               LK_legal = np.append(LK_legal, False)
            else:
                LK_legal = np.append(LK_legal, True)
        return LK_legal


    def run_ep(self):
        score = 0
        done = False
        while not done:
            action = self.choose_action(self.state, self.global_step, round(self.max_global_steps*3/4))
            action_continuous, action_discrete = action
            next_state, reward, done, info = self.env.step(action)
            self.remember(self.state, action_continuous, action_discrete, reward, next_state, done)
            score += reward
            self.state = next_state

        self.global_step = next(self.global_counter)
        self.returns_list.append(score)
        print(f"Worker: {self.name} Score is {score}, global steps {self.global_step}/{self.max_global_steps}")
        self.state = self.env.reset()

        if self.max_global_steps/20 % (self.global_step+1) == 0 and self.global_step > 100:
            print(f'global counter: {self.global_step}/{self.max_global_steps} \n')
            elapsed_time = time.time() - self.start_time
            remaining_time = elapsed_time * (self.max_global_steps - self.global_step) / max(self.global_step, 1)
            print(f'elapsed time: {elapsed_time / 60} min \n remaining time {remaining_time / 60} min \n')
            running_avg = np.mean(self.returns_list[-100:])
            print(f'running average is {running_avg}')

    def remember(self, state, action_continuous, action_discrete, reward, new_state, done):
        done = 1 - done
        self.memory.add((state, action_continuous, action_discrete, reward, new_state, done))

    #@tf.function
    def update_global_parameters(self):
        with tf.device('/CPU:0'):
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

                QvaluesDQN = self.local_dqn_model([state, action_continuous])
                Qvalues_next = self.local_dqn_model([new_state, self.local_param_model(new_state)])
                target_dqn = QvaluesDQN.numpy()  # get the shape
                batch_index = np.arange(self.batch_size, dtype=np.int32)
                target_dqn[batch_index, action_discrete] = reward + self.gamma * np.max(Qvalues_next, axis=1) * done

                loss_dqn = tf.keras.losses.MSE(QvaluesDQN, target_dqn)

                    # get gradients of loss with respect to the param_model weights
            gradient_param = tape.gradient(loss_param, self.local_param_model.trainable_weights)
            gradient_dqn = tape.gradient(loss_dqn, self.local_dqn_model.trainable_weights)

            # update global nets
            self.global_optimizer_P.apply_gradients(zip(gradient_param,
                                                        self.global_network_P.trainable_weights))
            self.global_optimizer_dqn.apply_gradients(zip(gradient_dqn,
                                                        self.global_network_dqn.trainable_weights))
            # update local nets
            self.local_param_model.set_weights(self.global_network_P.get_weights())
            self.local_dqn_model.set_weights(self.global_network_dqn.get_weights())


