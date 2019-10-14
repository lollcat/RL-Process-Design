from P_DQN_Agent import Agent
import numpy as np
from DistillationSimulator import Simulator

env = Simulator()
agent = Agent(alpha=0.0001 , beta=0.001, n_discrete_actions=env.discrete_action_space.n,
              n_continuous_actions=env.continuous_action_space.shape[0], state_shape=env.observation_space.shape,
              batch_size=32)
agent.load_models()

done = False
state = env.reset()
score = 0
while done is False: # run an episode
    print(state)
    action = agent.best_action(state)
    action_continuous, action_discrete = action
    state, reward, done, info = env.step(action)
    score += reward

print(score)
print(env.sep_order)
print(env.split_order)