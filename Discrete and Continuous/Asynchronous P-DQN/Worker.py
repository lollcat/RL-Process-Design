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
    def __init__(self, name, global_network_P, global_network_dqn, global_counter,
                 env, max_global_steps, n_steps=20, gamma=0.99):
        self.name = name
        self.global_network_P = global_network_P
        self.global_network_dqn = global_network_dqn
        self.global_optimizer_P =
        self.global_optimizer_dqn =
        self.global_counter = global_counter
        self.env = env
        self.n_steps = n_steps
        self.gamma = gamma
        self.local_param_model= global_network_P.copy()
        self.local_dqn_model= global_network_dqn.copy()

    def run(self, coord):
        try:
            while not coord.should_stop():
                # Collect some experience
                experience, global_step = self.run_n_steps()

                # Update the global networks using local gradients
                self.update_global_parameters(experience)

                # Stop once the max number of global steps has been reached
                if self.max_global_steps is not None and global_step >= self.max_global_steps:
                    coord.request_stop()
                    return
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

    def run_n_steps(self):
        experience = []
        for _ in range(self.n_steps)
            state = env.reset()
            action = self.choose_action(state, 1, 2) # TODO need to edit eps greedy - these are currently filler values
            action_continuous, action_discrete = action
            next_state, reward, done, info = env.step(action)
            step = Step(state, action, reward, next_state, done)
            experience.append(step)
            score += reward
            state = next_state

            global_step = next(self.global_counter)
            if done:
                break
        return steps, global_step

    def update_global_parameters(self, experience):
        reward = 0
        accumulated_param_gradients = 0
        accumulated_dqn_gradients = 0
        if not steps[-1].done:
            state = steps[-1].state[np.newaxis, :]
            action_continuous_predict = self.local_param_model.predict(state)
            target = self.local_dqn_model.predict([state, action_continuous_predict])
        for step in reversed(steps):
            target = step.reward + self.gamma * reward
            state = step.state[np.newaxis, :]
            with tf.GradientTape(persistent=True) as tape:
                predict_param = self.local_param_model(state)
                Qvalues = self.local_dqn_model([state, predict_param])
                Qvalue = Qvalues[step.action_discrete]
                loss_param = - tf.reduce_sum(Qvalues)
                loss_dqn = (Qvalue - target)**2
                # get gradients of loss with respect to the param_model weights
            gradient_param = tape.gradient(loss_param, self.local_param_model.trainable_weights)
            gradient_dqn = tape.gradient(loss_dqn, self.local_dqn_model.trainable_weights)
            if accumulated_param_gradients == 0:
                accumulated_param_gradients = gradient_param
                accumulated_dqn_gradients = gradient_dqn
            else:
                accumulated_param_gradients = [tf.add(accumulated_param_gradients[i], gradient_param[i])
                                               for i in range(len(gradient_param))]
                accumulated_dqn_gradients = [tf.add(accumulated_dqn_gradients[i], gradient_dqn[i])
                                               for i in range(len(gradient_dqn))]
        #update global nets
        self.global_optimizer_P.apply_gradients(zip(accumulated_param_gradients,
                                                    self.global_network_P.trainable_weights))
        self.global_optimizer_dqn.apply_gradients(zip(accumulated_dqn_gradients,
                                                    self.global_network_dqn.trainable_weights))
        #update local nets
        self.local_param_model.set_weights(self.global_network_P.get_weights())
        self.local_dqn_model.set_weights(self.global_network_dqn.get_weights())
    return












