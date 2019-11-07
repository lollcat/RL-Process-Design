import numpy as np
import tensorflow as tf
from Utils.OrnsteinNoise import OUActionNoise
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
                 global_counter, env, max_global_steps, returns_list, multiple_explore=False, freeze=False,
                 n_steps=10, gamma=0.99):
        self.name = name
        self.global_network_P = global_network_P
        self.global_network_dqn = global_network_dqn
        self.global_optimizer_P = global_optimizer_P
        self.global_optimizer_dqn = global_optimizer_dqn
        self.global_counter = global_counter
        self.env = env
        self.allow_submit = env.allow_submit
        self.multiple_explore = multiple_explore
        self.freeze = freeze
        self.state = self.env.reset()
        self.max_global_steps = max_global_steps
        self.global_step = 0
        self.returns_list = returns_list
        self.n_steps = n_steps
        self.gamma = gamma
        self.noise = OUActionNoise(mu=np.zeros(env.continuous_action_space.shape[0]))
        self.n_discrete_actions = env.discrete_action_space.n
        self.start_time = time.time()

        self.local_param_model = clone_model(global_network_P)
        self.local_param_model.set_weights(global_network_P.get_weights())
        self.local_dqn_model = clone_model(global_network_dqn)
        self.local_dqn_model.set_weights(global_network_dqn.get_weights())

    def run(self, coord):
        try:
            while not coord.should_stop():
                # Collect some experience
                experience = self.run_n_steps()
                # Update the global networks using local gradients
                self.update_global_parameters(experience)
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
        if self.freeze is False:
            noise = self.noise()
            mu_prime = mu + noise
            action_continuous = min(mu_prime, np.array([[0.999]]))  # cannot have a split above 1
        else:
            action_continuous = mu

        # get discrete action
        predict_discrete = self.local_dqn_model.predict([state, action_continuous])
        if self.multiple_explore is False:
            action_discrete = self.eps_greedy_action(state, predict_discrete, current_step, stop_step)
        else:
            if self.name % 3 == 0:
                action_discrete = self.diff_sep_explore(state, predict_discrete, current_step, stop_step)
            elif self.name % 3 == 1:
                action_discrete = self.eps_greedy_action(state, predict_discrete, current_step, stop_step)
            else:
                action_discrete = self.proportion_action(state, predict_discrete, current_step, stop_step)

        action_continuous = action_continuous[0]  # take it back to the correct shape
        return action_continuous, action_discrete

    def eps_greedy_action(self, state, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0.1):
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

    def diff_sep_explore(self, state, predict_discrete, current_step, stop_step, max_prob=1, min_prob=0):
        explore_threshold = max(max_prob - current_step / stop_step * (max_prob - min_prob), min_prob)
        random = np.random.rand()
        illegal_actions = self.illegal_actions(state)
        prev_LKs = self.env.sep_order
        non_explore_actions = np.zeros((self.n_discrete_actions,), dtype=bool)
        for i in range(len(prev_LKs)):
            non_explore_actions[np.arange(self.n_discrete_actions) % 5 == prev_LKs[i]] = True
        if self.allow_submit is True:
            non_explore_actions[-1] = False
        illegal_actions = illegal_actions + non_explore_actions
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

        non_explore_actions = np.zeros((self.n_discrete_actions,), dtype=bool)
        prev_LKs = self.env.sep_order
        for i in range(len(prev_LKs)):
            non_explore_actions[np.arange(self.n_discrete_actions) % 5 == prev_LKs[i]] = True
        if self.allow_submit is True:
            if self.env.n_outlet_streams > 1:
               LK_legal = np.append(LK_legal, False)
            else:
                LK_legal = np.append(LK_legal, True)
        LK_legal = LK_legal + non_explore_actions
        return LK_legal


    def run_n_steps(self):
        experience = []
        score = 0
        for _ in range(self.n_steps):
            action = self.choose_action(self.state, self.global_step, round(self.max_global_steps))
            action_continuous, action_discrete = action
            next_state, reward, done, info = self.env.step(action)
            step = Step(self.state, action_continuous, action_discrete, reward, next_state, done)
            experience.append(step)
            score += reward
            self.state = next_state
            self.global_step = next(self.global_counter)

            if done:
                self.returns_list.append(score)
                print(f"Worker: {self.name} Score is {score}, global steps {self.global_step}/{self.max_global_steps}")
                self.state = self.env.reset()
                break

            if self.max_global_steps/20 % (self.global_step+1) == 0 and self.global_step > 100:
                print(f'global counter: {self.global_step}/{self.max_global_steps} \n')
                elapsed_time = time.time() - self.start_time
                remaining_time = elapsed_time * (self.max_global_steps - self.global_step) / max(self.global_step, 1)
                print(f'elapsed time: {elapsed_time / 60} min \n remaining time {remaining_time / 60} min \n')
                running_avg = np.mean(self.returns_list[-100:])
                print(f'running average is {running_avg}')

        return experience

    #@tf.function
    def update_global_parameters(self, experience):
        with tf.device('/CPU:0'):
            target = 0
            accumulated_param_gradients = 0
            accumulated_dqn_gradients = 0
            if not experience[-1].done:
                state = experience[-1].state[np.newaxis, :]
                action_continuous_predict = self.local_param_model.predict(state)
                target = np.max(self.local_dqn_model.predict([state, action_continuous_predict]))
            for step in reversed(experience):
                target = step.reward + self.gamma * target
                state = step.state[np.newaxis, :]
                action_continuous = step.action_continuous[np.newaxis, :]
                action_discrete = step.action_discrete


                if self.freeze is False:
                    gradient_param, gradient_dqn = self.get_gradient(state, target, action_continuous, action_discrete)
                    if accumulated_dqn_gradients == 0:
                        accumulated_param_gradients = gradient_param
                        accumulated_dqn_gradients = gradient_dqn
                    else:
                        accumulated_param_gradients = [tf.add(accumulated_param_gradients[i], gradient_param[i])
                                                       for i in range(len(gradient_param))]
                        accumulated_dqn_gradients = [tf.add(accumulated_dqn_gradients[i], gradient_dqn[i])
                                                     for i in range(len(gradient_dqn))]
                else:
                    gradient_dqn = self.get_gradientDQN(state, target, action_continuous, action_discrete)

                    if accumulated_dqn_gradients == 0:
                        accumulated_dqn_gradients = gradient_dqn
                    else:
                        accumulated_dqn_gradients = [tf.add(accumulated_dqn_gradients[i], gradient_dqn[i])
                                                     for i in range(len(gradient_dqn))]

            self.update_all_weights(accumulated_dqn_gradients, accumulated_param_gradients)
            return

    #@tf.function
    def get_gradient(self, state, target, action_continuous, action_discrete):
        with tf.GradientTape(persistent=True) as tape:
            # param part
            predict_param = self.local_param_model(state)
            Qvalues = self.local_dqn_model([state, predict_param])
            loss_param = - tf.reduce_sum(Qvalues)

            # dqn part
            QvaluesDQN = self.local_dqn_model([state, action_continuous])
            target_dqn = QvaluesDQN.numpy()
            target_dqn[:, action_discrete] = target
            target_dqn = tf.convert_to_tensor(target_dqn)
            loss_dqn = tf.keras.losses.MSE(QvaluesDQN, target_dqn)
            # get gradients of loss with respect to the param_model weights
        gradient_param = tape.gradient(loss_param, self.local_param_model.trainable_weights)
        gradient_param = [tf.clip_by_norm(grad, 5) for grad in gradient_param]

        gradient_dqn = tape.gradient(loss_dqn, self.local_dqn_model.trainable_weights)
        gradient_dqn = [tf.clip_by_norm(grad, 5) for grad in gradient_dqn]

        return gradient_param, gradient_dqn

    def get_gradientDQN(self, state, target, action_continuous, action_discrete):
        with tf.GradientTape(persistent=True) as tape:
            QvaluesDQN = self.local_dqn_model([state, action_continuous])
            target_dqn = QvaluesDQN.numpy()
            target_dqn[:, action_discrete] = target
            target_dqn = tf.convert_to_tensor(target_dqn)
            loss_dqn = tf.keras.losses.MSE(QvaluesDQN, target_dqn)
        gradient_dqn = tape.gradient(loss_dqn, self.local_dqn_model.trainable_weights)
        gradient_dqn = [tf.clip_by_norm(grad, 5) for grad in gradient_dqn]
        return gradient_dqn

    #@tf.function
    def update_all_weights(self, accumulated_dqn_gradients, accumulated_param_gradients):
        self.global_optimizer_dqn.apply_gradients(zip(accumulated_dqn_gradients,
                                                      self.global_network_dqn.trainable_weights))
        self.local_dqn_model.set_weights(self.global_network_dqn.get_weights())
        if accumulated_param_gradients is 0:
            return
        else:
            # update global nets
            self.global_optimizer_P.apply_gradients(zip(accumulated_param_gradients,
                                                        self.global_network_P.trainable_weights))
            # update local nets
            self.local_param_model.set_weights(self.global_network_P.get_weights())
            return
