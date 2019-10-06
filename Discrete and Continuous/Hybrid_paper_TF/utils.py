import matplotlib.pyplot as plt
import numpy as np
from os import times


class Plotter(object):
    def __init__(self, score_history, episodes):
        self.score_history = score_history
        self.episodes = episodes + 1

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    def plot(self, save=False):
        episodes = np.arange(self.episodes)
        smoothed_rews = self.running_mean(self.score_history, 100)
        plt.plot(episodes[-len(smoothed_rews):], smoothed_rews)
        plt.plot(episodes, self.score_history, color='grey', alpha=0.3)
        plt.xlabel("steps")
        plt.ylabel("reward")
        plt.legend(["avg reward", "reward"])
        if save is True:
            plt.savefig("RewardvsSteps_" + str(os.times().user) + ".png")
        plt.show()
