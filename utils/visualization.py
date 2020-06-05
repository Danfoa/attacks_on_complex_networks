import os
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)


def plot_metric_distribution(metrics_quantiles, ratios, y_label, labels, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    colors = cm.ocean(np.linspace(0.0, 0.7, len(metrics_quantiles)))

    for quantiles, c in zip(metrics_quantiles, colors):
        ax.plot(ratios, quantiles[:, 1], '-', color=c)
        ax.fill_between(ratios, quantiles[:, 0], quantiles[:, 2], color=c, alpha=0.15)

    ax.grid("on", alpha=0.1)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Removal ratio")
    ax.set_title(title)
    if labels:
        ax.legend(labels).set_zorder(10)

    plt.tight_layout()
    if not os.path.exists(os.path.join('results')):
        os.makedirs(os.path.join('results'))
    filename = os.path.join('results', filename)

    plt.savefig(filename)


def plot_clustering_distribution(metrics_clusterings, ratios, y_label, labels, title, filename):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    avg_size_isolated_clusters,  relative_size_largest_cluster = zip(*metrics_clusterings)
    metrics_quantiles = [list(avg_size_isolated_clusters), list(relative_size_largest_cluster)]

    colors = cm.ocean(np.linspace(0.0, 0.7, len(metrics_quantiles)))
    markers = ["s", "o", "s", "o"]

    for quantiles, c, marker in zip(metrics_quantiles, colors, markers):
        ax.plot(ratios, quantiles, marker, fillstyle='none', color=c)

    ax.grid("on", alpha=0.1)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Removal ratio")
    ax.set_title(title)
    if labels:
        ax.legend(labels).set_zorder(10)

    plt.tight_layout()
    filename = os.path.join('results', filename)
    plt.savefig(filename)


def save_network_tracking(network_tracking, title, file_name):
    ratios = sorted(list(network_tracking.keys()))
    num_plots = len(ratios)

    for i, ratio in enumerate(ratios):
        filename = os.path.join("results", file_name, "network_tracking({}).net".format(i))
        nx.write_pajek(network_tracking[ratio], filename)
