from keras import backend as K #need for 
from keras.layers import Dense, Input, Lambda, LSTM, Reshape
from keras.models import Model
from keras.optimizers import Adam
import numpy as np

"""
Used for help: 
General Format"
https://www.youtube.com/watch?v=2vJtbAha3To, 
https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/actor_critic/actor_critic_keras.py 

DPG:
https://github.com/germain-hug/Deep-RL-Keras/blob/master/DDPG/actor.py
"""

class Agent(object):
    def __init__(self, env, alpha, beta, gamma, layer_size = 20):
        
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.layer_size = layer_size
        self.act_range = (env.continuous_action_space.high-env.continuous_action_space.low)
        self.act_min = env.continuous_action_space.low
        
        self.actor, self.critic, self.policy = self.build_actor_critic_network()
    
    def build_actor_critic_network(self):
        input = Input(shape=self.env.observation_space.shape, name = "Input")
        delta = Input(shape = [1]) #related to calculation of loss function
        
        
        dense1 = Dense(self.layer_size, activation = 'relu', name = "fc1")(input)
        dense2 = Dense(self.layer_size, activation = 'relu', name = "fc2")(dense1)
        reshape = Reshape((self.layer_size,1))(dense2)
        lstm = LSTM(self.layer_size)(reshape)
        
        
        policy_discrete_probs = Dense(self.env.discrete_action_space.n, activation = 'softmax', name = "discrete_action_output")(lstm)
        policy_continuous = Dense(self.env.continuous_action_space.shape[0], activation = 'sigmoid', name = "policy_output")(lstm)
        policy_continuous = Lambda(lambda i: i*self.act_range+self.act_min, name = "policy_output_actual_value")(policy_continuous)
        
        Values = Dense(1, activation = 'linear', name = "Critic_output")(lstm) # Critic outputs a single value 
    
        def custom_loss(y_true, y_pred): 
            """
            keras requires y_true, y_pred format for custom loss
            y_true is the actions the agent took, y_pred is the NN output 
            y_true  is representation of action that was taken
            delta is related to output of the critic network
            """
            out = K.clip(y_pred[0], 1e-8, 1-1e-8) #need to clip to prevent potentially taking log 0
            log_lik = y_true[0]*K.log(out) 
            
            return K.sum(-log_lik*delta)
        
        actor = Model(input = [input, delta], output = [policy_discrete_probs, policy_continuous])
        actor.compile(optimizer = Adam(lr=self.alpha), loss = custom_loss)
        
        critic = Model(input = [input], output = [Values])
        critic.compile(optimizer = Adam(lr = self.beta), loss = 'mse')
        
        #need to get policy (don't take critic delta input when selecting action, need seperate thingy just to calculate feed forward of the network
        #don't need to compile because there is no backprop
        policy = Model(input = [input], output = [policy_discrete_probs, policy_continuous])        
        return actor, critic, policy
        
    def choose_action_random(self, observation):
        state = observation[np.newaxis, :]
        policy_discrete_probs, policy_continuous_predict = self.policy.predict(state)
        policy_discrete_probs = policy_discrete_probs[0]
        
        action_discrete = np.random.choice(self.env.discrete_action_space.n, p = policy_discrete_probs)    
            
        Noise = np.random.normal(policy_continuous_predict, 0.2)
        action_continuous = policy_continuous_predict + Noise
        
        
        return action_discrete, action_continuous
    
    
    def choose_action_epsgreedy(self, observation, current_step, stop_step, max_prob = 0.95, min_prob = 0.05, max_noise = 0.95, min_noise = 0.01):
        
        
        state = observation[np.newaxis, :]
        policy_discrete_probs, policy_continuous_predict = self.policy.predict(state)      
            
        
        explore_threshold = max(max_prob - current_step/stop_step * (max_prob - min_prob), min_prob)
        Noise_sd = max(max_noise - current_step/stop_step * (max_noise - min_noise), min_noise) 
        
        Noise = np.random.normal(policy_continuous_predict, Noise_sd)
        action_continuous = min(self.env.continuous_action_space.high.reshape(policy_continuous_predict.shape), policy_continuous_predict + Noise)
        random = np.random.rand()
        if random > explore_threshold:
            action_discrete = np.random.choice(self.env.discrete_action_space.n)    
        else:
           action_discrete = np.argmax(policy_discrete_probs) 
        

        
        return action_discrete, action_continuous
        
    
    def learn(self, state, action, reward, next_state, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]
        
        action_discrete, action_continuous = action
        
    
        critic_value = self.critic.predict(state)
        critic_value_next = self.critic.predict(next_state)
        
        if done == False:
            target = reward + self.gamma*critic_value_next
        if done == True:
            target = np.array(reward).reshape((1,1))
        
        delta = target - critic_value
        
        actions_discrete = np.zeros([1, self.env.discrete_action_space.n])
        actions_discrete[np.arange(1), action_discrete] = 1.0
    
        self.actor.fit([state, delta], [actions_discrete, action_continuous], verbose = 0) #verbose = 0 stops outputs from displaying
        self.critic.fit(state, target, verbose = 0)
        
        
#agent = Agent(env = simulator(), alpha = 1e-4, beta = 5e-4, gamma = 0.99)
        
