from tensorflow.keras.backend import set_floatx
set_floatx('float64')
from P_DQN_Agent import Agent
import numpy as np
from utils import Plotter
from DistillationSimulator import Simulator
import time

env = Simulator()
agent = Agent(alpha=0.0001, beta=0.001, n_discrete_actions=env.discrete_action_space.n,
              n_continuous_actions=env.continuous_action_space.shape[0], state_shape=env.observation_space.shape,
              batch_size=32)
np.random.seed(0)
total_eps = 1000
total_eps_greedy = total_eps/2

# if there is a saved agent then uncomment below:
agent.load_models()

score_history = []
start_time = time.time()
for i in range(total_eps):
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = agent.choose_action(state, i, total_eps_greedy)
        action_continuous, action_discrete = action
        new_state, reward, done, info = env.step(action)
        agent.remember(state, action_continuous, action_discrete, reward, new_state, int(done))
        agent.learn()
        score += reward
        state = new_state
        # env.render()

    score_history.append(score)
    if i % 100 == 0:
        print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))
        elapsed_time = (time.time() - start_time)/60
        print(f'elapsed_time is {elapsed_time} min \n')
        time_left = (total_eps - i)/(i+1) * elapsed_time
        print(f'estimated time to go is {time_left/60} hr \n')

    if i % (total_eps/20) == 0 and i > 100:
        env.render()
        plotter = Plotter(score_history, i)
        plotter.plot()
        elapsed_time = (time.time() - start_time) / 60
        print(f'elapsed_time is {elapsed_time} min \n')
        time_left = (total_eps - i) / (i + 1) * elapsed_time
        print(f'estimated time to go is {time_left / 60} hr \n')

agent.save_models()

plotter = Plotter(score_history, total_eps-1)
plotter.plot(save=True)
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
