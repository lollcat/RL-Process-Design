from Env.Simulator_New import Simulator
import numpy as np

env = Simulator()
reward_list = []
for i in range(100000):
    env.reset()
    done = False
    score = 0
    while not done:
        state, reward, done, _ = env.run_random()
        score += reward

    reward_list.append(score)

average = np.average(reward_list)
print(average)
