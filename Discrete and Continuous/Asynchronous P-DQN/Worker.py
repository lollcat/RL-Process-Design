from DQN import DQN_Agent
from P_actor import ParameterAgent
import numpy as np
import tensorflow as ft


class Step:  # Stores a step
    def __init__(self, state, action_continuous, action_discrete, reward, next_state, done):
        self.state = state
        self.action_continuous = action_continuous
        self.action_discrete = action_discrete
        self.reward  = reward
        self.next_state = next_state
        self.done = done


def Worker:
    def __init__(self, global_network_P, global_network_DQN, global_counter, env, max_global_steps, gamma=0.99)
        self.global_network_P = global_network_P
        self.global_network_DQN = global_network_DQN
        self.global_counter = global_counter
        self.env = env

        self.local_param_model= global_network_P.copy()
        self.local_dqn_model= global_network_DQN.copy()

    def run(self, coord, t_max):
        try:
            while not coord.should_stop():
                # Copy weights from  global networks to local networks
                self.update_local_parameters()

                # Collect some experience
                steps, global_step = self.run_n_steps(t_max)

                # Stop once the max number of global steps has been reached
                if self.max_global_steps is not None and global_step >= self.max_global_steps:
                    coord.request_stop()
                    return

                # Update the global networks using local gradients
                self.update(steps, sess)

        except tf.errors.CancelledError:
            return

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
        predict_discrete = self.loal_dqn_model.predict([state, action_continuous])
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

    def run_n_steps(self, n):
        steps = []
        for _ in range(n)
            state = env.reset()
            action = self.choose_action(state, i, total_eps_greedy) # TODO need to edit eps greedy
            action_continuous, action_discrete = action
            next_state, reward, done, info = env.step(action)
            step = Step(state, action, reward, next_state, done)
            steps.append(step)
            score += reward
            state = new_state

            global_step = next(self.global_counter)
            if done:
                break
            return steps, global_step

    def update(self, steps):





