import os
from Distillation_disc_cont import Simulator
import numpy as np
from PDQN_Agent import Agent
from utils import plotLearning
import matplotlib.pyplot as plt


env = Simulator()
agent = Agent(alpha=0.00001, beta=0.000001, input_dims=env.observation_space.shape, tau=0.001,
              env=env, batch_size=64, layer1_size=800, layer2_size=600,
              n_discrete_actions=env.discrete_action_space.n, n_continuous_actions=1)
np.random.seed(0)
total_eps = 10000
# if there is a saved agent then uncomment below:
#agent.load_models()

score_history = []
for i in range(total_eps):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state, i, total_eps)
        action_continuous, action_discrete = action
        new_state, reward, done, info = env.step(action)
        agent.remember(state, action_continuous, action_discrete, reward, new_state, int(done))
        agent.learn()
        score += reward
        state = new_state
        #env.render()

    score_history.append(score)
    if i%50 == 0:
        print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

    if i%1000 == 0:
        pass
        #agent.save_models()

#agent.save_models()


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N

episodes = np.arange(total_eps)
smoothed_rews = running_mean(score_history, 100)
plt.plot(episodes[-len(smoothed_rews):], smoothed_rews)
plt.plot(episodes, score_history,color='grey', alpha=0.3)
plt.xlabel("steps")
plt.ylabel("reward")
plt.legend(["avg reward", "reward"])
plt.show()

"""
done = False
state = env.reset()
while done is False: # run an episode
    action = agent.best_action(state)
    action = action_continuous, action_discrete
    new_state, reward, done, info = env.step(action)
"""