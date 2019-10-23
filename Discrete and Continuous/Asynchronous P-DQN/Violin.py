
from Env.Simulator_New import Simulator
import matplotlib.pyplot as plt

env = Simulator()
profit = []
for _ in range(1000):
    done = False
    while not done:
        state, reward, done, _ = env.run_random()
    profit.append(env.Profit)
    env.reset()

plt.plot(profit)
plt.show()