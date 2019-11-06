from Env.Simulator_New import Simulator
import numpy as np

env = Simulator()
reward_list1 = []
reward_list2 = []
for i in range(10000):
    env.reset()
    done = False
    score = 0
    while not done:
        state, reward, done, _ = env.run_random()
        score += reward
    reward_list1.append(score)
    reward_list2.append(env.Performance_metric2)

average1 = np.average(reward_list1)
print(average1)
average2 = np.average(reward_list2)
print(average2)

