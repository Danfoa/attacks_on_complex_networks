import os
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from utils.data_saver import load_results

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
    cols = int((num_plots + 1) / 2)
    size = 5
    plt.figure(figsize=(cols * size, 2 * size))
    pos = nx.random_layout(network_tracking[ratios[0]])
    for i, ratio in enumerate(ratios):
        filename = os.path.join("results", file_name, "network_tracking({}).net".format(i))
        nx.write_pajek(network_tracking[ratio], filename)
        plt.subplot(2, cols, i + 1)
        nx.draw(network_tracking[ratio], node_size=10, pos=pos)
        plt.title("Ratio:{} - N:{}".format(ratio, network_tracking[ratio].number_of_nodes()))

    plt.savefig("../results/{}_network_tracking.png".format(file_name))


def plot_comparisons_from_file_metrics(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate):
    steps = int(exp_max_rate / exp_removal_rate)
    xaxis = np.cumsum([exp_removal_rate] * steps)

    colors = cm.ocean(np.linspace(0.0, 0.7, len(filenames)))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    min_paths, max_paths, cluster_size_ratios, ylabels = [], [], [], []
    for i, f in enumerate(filenames):
        min_path, max_path, cluster_size_ratios = load_results(f)

        for quantiles in [min_path]:#, max_path]:
            c = colors[i]
            ax.plot(xaxis, quantiles[:, 1], '-', color=c)
            ax.fill_between(xaxis, quantiles[:, 0], quantiles[:, 2], color=c, alpha=0.15)

        ylabels.append(r'$d_{min}: $' + labels[i])
        #ylabels.append(r'$d_{max}: $' + labels[i])

    ax.grid("on", alpha=0.1)
    ax.set_ylabel("path length")
    ax.set_xlabel("Removal ratio")
    ax.set_title(title)
    if labels:
        ax.legend(ylabels).set_zorder(10)

    plt.tight_layout()
    file_name = os.path.join('results', file_name + "_dia_comparisons")
    plt.savefig(file_name)


def plot_comparisons_from_file_clustering(file_name, title, filenames, labels, exp_max_rate, exp_removal_rate):

    steps = int(exp_max_rate / exp_removal_rate)
    xaxis = np.cumsum([exp_removal_rate] * steps)

    colors = cm.ocean(np.linspace(0.0, 0.7, len(filenames)))
    markers_all = ["s","x","o",".","D","*"]
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    min_paths, max_paths, cluster_size_ratios, ylabels = [], [], [], []
    c = 0
    for i,f in enumerate(filenames):
        min_path, max_path, cluster_size_ratios = load_results(f)

        clusters_info = [x[0] for x in cluster_size_ratios]
        avg_size_isolated_clusters, relative_size_largest_cluster = zip(*clusters_info)
        metrics_quantiles = [list(avg_size_isolated_clusters), list(relative_size_largest_cluster)]

        markers = [(2 * i + j)%len(markers_all) for j in range(2)]
        markers = [markers_all[j] for j in markers]

        #for quantiles, marker in zip(metrics_quantiles, markers):
        ax.plot(xaxis, metrics_quantiles[0], "-", fillstyle='none', color=colors[i])
        ax.plot(xaxis, metrics_quantiles[1], '--', fillstyle='none', color=colors[i])

        ylabels.append("S: "+labels[i])
        ylabels.append("<s>: "+labels[i])

    ax.grid("on", alpha=0.1)
    ax.set_ylabel("clusterings")
    ax.set_xlabel("Removal ratio")
    ax.set_title(title)
    if labels:
        ax.legend(ylabels).set_zorder(10)

    plt.tight_layout()
    file_name = os.path.join('results', file_name+"_clust_comparisons")
    plt.savefig(file_name)

