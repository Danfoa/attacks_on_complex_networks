
import os
import copy
import glob
import numpy as np
from math import log, ceil
from functools import reduce  # forward compatibility for Python 3
import operator

import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx





def plot_network_communities(nets, titles, number_of_clusters, save_path=None):
    """
    Function to plot different community detection algorithms side by side
    """

    original_position = nx.spring_layout(nets[0], center=[0, 0], scale=1)
    # original_position = nx.random_layout(nets[0], center=[-0.5, -0.5])

    cols = 1
    rows = int(len(nets ) /cols)

    plt.figure(figsize=(5.0 * cols, 5.0 * rows), dpi=120)

    for i, net in enumerate(nets):
        print(i)
        ax = plt.subplot(rows, cols, i + 1)
        components = [c for c in nx.connected_components(net) if len(c) > 1]
        components = sorted(components, key=len, reverse=True)
        components = components[:number_of_clusters] if len(components) > number_of_clusters else components

        colors = cm.ocean(np.linspace(0.75, 0, len(components)))
        colors[0] = [0.047, 0.360, 0.392, 1]

        angles = np.linspace(0, 360 / 180 * math.pi, len(components))
        new_pos = copy.deepcopy(original_position)

        for n, subnet_nodes, color in zip(range(len(components)), components, colors):
            if n != 0:
                subnet_cord = {node: new_pos.get(node) for node in subnet_nodes}
                coordinates = np.array(list(subnet_cord.values()))
                scale = 1  # max(0.5, len(subnet_nodes)/len(components[0]))
                emp_mean = np.mean(coordinates, axis=0)
                empirical_diameter = np.max(coordinates, axis=0) - np.min(coordinates, axis=0)

                for node in subnet_nodes:
                    x = (new_pos[node][0] - emp_mean[0]) * scale - \
                                ((1 + np.linalg.norm(empirical_diameter)) * np.cos(angles[n - 1]))
                    y = (new_pos[node][1] - emp_mean[1]) * scale + (
                                (1 + np.linalg.norm(empirical_diameter)) * np.sin(angles[n - 1]))
                    new_pos[node] = [x, y]

            nx.draw(net.subgraph(subnet_nodes),
                    node_size=6, width=0.25,
                    node_color=np.expand_dims(color, axis=0),
                    with_labels=False,
                    pos=new_pos,
                    linewidths=None,
                    ax=ax)
            ax.set_title(titles[i])

        p = [plt.Circle((0, 0), 0.2, fc=c) for c in colors]
        ax.legend(p, ["%.3f" % (len(n) / len(original_position)) for n in components])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    # Example code to plot tracing of clusters
    exp_path = "results/attack_nodes_3000/powerlaw-incr-attack-k=3.20-n_nodes=3000"

    # Get the list of networks.
    nets = []
    for file in os.scandir(exp_path):
        if ".net" in file.path:  # Load Network
            print(file.name)
            # Load net
            nets.append(nx.read_pajek(file.path))

    nets = [nets[0], nets[2], nets[4], nets[8], nets[-1]]

    plot_network_communities(nets, ["Title"] * len(nets), number_of_clusters=7
)