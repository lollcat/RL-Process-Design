import matplotlib.pyplot as plt
import numpy as np
from os import times
from IPython.display import Image
import pydot
import imageio

class Plotter:
    def __init__(self, score_history, episodes):
        self.score_history = score_history
        self.episodes = episodes + 1
        if episodes < 100:
            raise ValueError("Not enough episodes")

    def running_mean(self, x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    def plot(self, save=False):
        episodes = np.arange(self.episodes)
        smoothed_rews = self.running_mean(self.score_history, 100)
        plt.plot(episodes[-len(smoothed_rews):], smoothed_rews)
        plt.plot(episodes, self.score_history, color='grey', alpha=0.3)
        plt.xlabel("episodes")
        plt.ylabel("reward")
        plt.legend(["avg reward", "reward"])
        if save is True:
            plt.savefig("RewardvsSteps_" + str(times().user) + ".png")
        plt.show()

class Visualiser:
    def __init__(self, env):
        self.env = env

    def visualise(self):
        env = self.env
        G = pydot.Dot(graph_type="digraph", rankdir="LR")
        outlet_nodes = []
        nodes = []
        edges = []
        image_list = []

        for i in range(len(env.sep_order)):
            LK = env.sep_order[i]
            split = env.split_order[i].round(3)[0]
            nodes.append(pydot.Node(f'Column {i + 1} \nLK is {LK} \nsplit is {split}', shape="square"))
            G.add_node(nodes[i])
            if i > 0:
                stream_in = env.column_streams[i][0]
                column_link, loc = self.find_column(stream_in)
                edges.append(pydot.Edge(nodes[column_link], nodes[i], headport="w", tailport=loc))
                G.add_edge(edges[i - 1])

            # add outlet streams
            tops, bottoms = env.column_streams[i][1:]
            stream = env.stream_table[tops]
            flowrate = int(stream.sum())
            purity = int(100 * stream.max() / stream.sum())
            compound = stream.argmax()
            compound = env.compound_names[compound]
            outlet_nodes.append(
                pydot.Node(f"Tops Column {i+1} \n {flowrate} kmol/s \n{purity}% {compound}", shape="box", color="white"))
            G.add_node(outlet_nodes[-1])
            G.add_edge(pydot.Edge(nodes[i], outlet_nodes[-1], headport="w", tailport="ne"))

            stream = env.stream_table[bottoms]
            flowrate = int(stream.sum())
            purity = int(100 * stream.max() / stream.sum())
            compound = stream.argmax()
            compound = env.compound_names[compound]
            outlet_nodes.append(pydot.Node(f"Bottoms Column {i+1} \n {flowrate} kmol/s \n{purity}% {compound} {i}", shape="box", color="white"))
            G.add_node(outlet_nodes[-1])
            G.add_edge(pydot.Edge(nodes[i], outlet_nodes[-1], headport="w", tailport="se"))
        BFD = imageio.imread(G.create_png())
        return BFD


    def find_column(self, stream):
        env = self.env
        for i in range(len(env.column_streams)):
            if stream in env.column_streams[i]:
                if env.column_streams[i][1] == stream:
                    loc = "ne"
                    return i, loc
                elif env.column_streams[i][2] == stream:
                    loc = "se"
                    return i, loc
                else:
                    print("error")
