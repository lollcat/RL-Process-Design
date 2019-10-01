from keras import backend as K
from keras.layers import Dense, Input, Lambda, Add
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from scipy.special import softmax

"""
Used for help: 
General Format"
https://www.youtube.com/watch?v=2vJtbAha3To, 
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/actor_critic/actor_critic_keras.py 

DPG:
https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
"""

class Agent(object): #  todo make class like that hybrid paper

    def __init__(self, env, alpha, beta, gamma, layer_size = 20):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.layer_size = layer_size
        self.act_range = (env.continuous_action_space.high-env.continuous_action_space.low)
        self.act_min = env.continuous_action_space.low
        
        self.actor_param, self.actor_DQN = self.build_P_DQN()

    def build_P_DQN(self):
        input_state = Input(shape=self.env.observation_space.shape, name = "Input")
        
        #Define Parameter NN
        dense1_param = Dense(self.layer_size, activation = 'relu', name = "fc1")(input_state)
        dense2_param = Dense(self.layer_size, activation = 'relu', name = "fc2")(dense1_param)
        #dense3_param = Dense(self.layer_size, activation = 'relu', name = "fc3")(dense2_param)
        policy_continuous = Dense(self.env.continuous_action_space.shape[0], activation = 'sigmoid', name = "policy_output")(dense2_param)
        policy_continuous = Lambda(lambda i: i*self.act_range+self.act_min, name = "policy_output_actual_value")(policy_continuous)
        
        #Define DQN
        input_state_DQN = Input(shape=self.env.observation_space.shape, name = "Input_state_DQN")
        dense1_state_DQN = Dense(self.layer_size, activation = 'relu', name = "dense1_state_DQN")(input_state_DQN)
        input_param_DQN = Input(shape = self.env.continuous_action_space.shape, name = "input_param_DQN")
        dense1_param_DQN = Dense(self.layer_size, activation = 'relu', name = "dense1_param_DQN")(input_param_DQN)
        
        input_DQN = Add()([dense1_state_DQN, dense1_param_DQN])
        dense1_DQN = Dense(self.layer_size, activation='relu', name = "dense1_DQN")(input_DQN)
        dense2_DQN = Dense(self.layer_size, activation = 'relu', name = "dense2_DQN")(dense1_DQN)

        policy_discrete_probs = Dense(self.env.discrete_action_space.n, activation = 'linear', name = "discrete_action_output")(dense2_DQN)

        def custom_loss(y_true, y_pred):
            return -y_true #where y_true is the Q value
            
        actor_param = Model(input = input_state, output = policy_continuous)
        actor_param.compile(optimizer = Adam(lr=self.alpha), loss = custom_loss)
        
        actor_DQN = Model(input = [input_state_DQN, input_param_DQN], output = [policy_discrete_probs])
        actor_DQN.compile(optimizer = Adam(lr = self.beta), loss = 'mse')
        
        return actor_param, actor_DQN
        
    def choose_action_random(self, observation):
        state = observation[np.newaxis, :]

        policy_continuous_predict = self.actor_param(state)
        policy_discrete_probs = softmax(self.actor_DQN(state, policy_continuous_predict))
        policy_discrete_probs = policy_discrete_probs[0]
        
        action_discrete = np.random.choice(self.env.discrete_action_space.n, p = policy_discrete_probs)    
            
        Noise = np.random.normal(policy_continuous_predict, 0.2)
        action_continuous = policy_continuous_predict + Noise
        
        
        return action_discrete, action_continuous
    
    
    def choose_action_epsgreedy(self, observation, current_step, stop_step, max_prob = 0.95, min_prob = 0.05, max_noise = 0.8, min_noise = 0.01):
        
        
        state = observation[np.newaxis, :]

        policy_continuous_predict = self.actor_param.predict(state)
        policy_discrete_probs = softmax(self.actor_DQN.predict([state, policy_continuous_predict]))
        policy_discrete_probs = policy_discrete_probs[0]    
            
        
        explore_threshold = max(max_prob - current_step/stop_step * (max_prob - min_prob), min_prob)
        Noise_sd = max(max_noise - current_step/stop_step * (max_noise - min_noise), min_noise) 
        
        Noisey_continuous = np.random.normal(policy_continuous_predict, Noise_sd)
        action_continuous = min(self.env.continuous_action_space.high.reshape(policy_continuous_predict.shape), Noisey_continuous)
        random = np.random.rand()
        if random > explore_threshold:
            action_discrete = np.random.choice(self.env.discrete_action_space.n)    
        else:
           action_discrete = np.argmax(policy_discrete_probs) 
        

        
        return action_discrete, action_continuous
        
    
    def learn(self, state, action, reward, next_state, done, batch_size, verbose = 0):
        reward = reward
        #assert reward.shape == (batch_size, 1), "wrong reward shape"
        state = state
        next_state = next_state
        assert state.shape == (batch_size, self.env.observation_space.shape[0]), "state shape wrong"
        
        action_discrete = action[:, 0].astype('int')
        assert action_discrete.shape == (batch_size,), "discrete action space wrong"
        
        action_continuous  = action[:, 1].reshape(batch_size, 1)
        #assert action_continuous.shape == (batch_size, 1), "continuous action space wrong"
        
        # Compute param loss function
        policy_continuous_predict = self.actor_param.predict(state)
        dqn_predict_given_param = self.actor_DQN.predict([state, policy_continuous_predict])
        loss_param = np.sum(dqn_predict_given_param, axis = 1)
        loss_param = loss_param.reshape((batch_size, 1))
        #assert loss_param.shape == (batch_size, 1), f"loss_param.shape wrong: {loss_param}"
        
        #compute DQN loss function
        #First compute Q value for the taken discrete action, given the current state and continuous action taken
        dqn_predict_given_continuous = self.actor_DQN.predict([state, action_continuous])
        values = np.zeros((batch_size, self.env.discrete_action_space.n))
        values[np.arange(batch_size), action_discrete] = dqn_predict_given_continuous[np.arange(batch_size), action_discrete]
        
        #Next compute the Q value of the next state given the next state and the NN weights for both networks
        policy_continuous_predict = self.actor_param.predict(next_state)
        dqn_predict_next = self.actor_DQN.predict([next_state, policy_continuous_predict])
        values_next = np.zeros((batch_size, self.env.discrete_action_space.n))
        values_next[np.arange(batch_size), action_discrete] = np.argmax(dqn_predict_next[np.arange(batch_size), action_discrete])
        
        reward_matrix = np.zeros((batch_size, self.env.discrete_action_space.n))
        reward_matrix[np.arange(batch_size), action_discrete] = reward
        
        if done == False:
            target = reward_matrix + self.gamma*values_next
        if done == True:
            target = reward_matrix
        
        assert target.shape == (batch_size, self.env.discrete_action_space.n), target
        #if target.shape != (batch_size, 1): target = target.reshape((batch_size, 1))
        
        
        
        #print(target.shape, critic_value.shape)
        #print(state.shape, delta.shape, actions_discrete.shape, action_continuous.shape)
        self.actor_param.fit(state, loss_param, verbose = 0, batch_size = batch_size) #verbose = 0 stops outputs from displaying
        self.actor_DQN.fit(state, target, verbose = 0, batch_size = batch_size)
        
        
#agent = Agent(env = Simulator(), alpha = 1e-4, beta = 5e-4, gamma = 0.99)
        
