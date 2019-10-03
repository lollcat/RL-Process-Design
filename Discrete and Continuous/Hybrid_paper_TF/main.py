import os
from Distillation_disc_cont import Simulator
import numpy as np
from PDQN_Agent import Agent
from utils import plotLearning


env = Simulator()
agent = Agent(alpha=0.0001, beta=0.00001, input_dims=env.observation_space.shape, tau=0.001,
              env=env, batch_size=64, layer1_size=800, layer2_size=600,
              n_discrete_actions=1, n_continuous_actions=1)
np.random.seed(0)
# if there is a saved agent then uncomment below:
# agent.load_models()

score_history = []
for i in range(5000):
    obs = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(obs)
        action_continuous, action_discrete = action
        new_state, reward, done, info = env.step(act)
        agent.remember(obs, action_continuous, action_discrete, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()

    score_history.append(score)
    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    if i%100 == 0: agent.save_models

filename = 'Pendulum-alpha00005-beta0005-800-600-optimized.png'
plotLearning(score_history, filename, window=100)