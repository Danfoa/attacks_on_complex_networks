import time
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from scipy.stats import poisson

from network_attacks import instantaneous_attack
from utils.configuration_model import ConfigurationGenerator

warnings.filterwarnings("ignore", category=UserWarning)


def get_power_law_net(n_nodes, k, verbose=False):
    distribution = powerlaw.Power_Law(xmin=1, parameters=[k])
    degrees = distribution.generate_random(n_nodes).astype(np.int32)

    start_time = time.time()
    generator = ConfigurationGenerator(degrees)
    net = generator.get_network()

    if verbose:
        print("** Gen PowerLaw k=%.2f took %.3fs" % (k, time.time() - start_time))

    return net


def get_poisson_net(n_nodes, mu, verbose=False):
    distribution = poisson(mu)
    degrees = distribution.rvs(size=n_nodes)

    start_time = time.time()
    generator = ConfigurationGenerator(degrees)
    net = generator.get_network()

    if verbose:
        print("** Gen Poisson mu=%.2f took %.3fs" % (mu, time.time() - start_time))

    return net


def plot_metric_distribution(metrics_quantiles, ratios, y_label, labels, title=''):
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
    plt.show()


def plot_clustering_distribution(metrics_clusterings, ratios, y_label, labels, title=''):
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
    plt.show()


if __name__ == "__main__":
    exp_removal_ratios = np.linspace(0.0, 0.5, 10)
    exp_num_nodes = [100]
    exp_mus = [4]
    exp_ks = [2.6]
    exp_max_rate = 0.5
    exp_removal_rate = 0.02
    for n_nodes in exp_num_nodes:
        for mu in exp_mus:
            net = get_poisson_net(n_nodes=n_nodes, mu=mu, verbose=False)
            attacked_net = net

            min_path, max_path, cluster_size_ratios = instantaneous_attack(net, exp_removal_ratios,  verbose=True)

            print("Final cluster sizes:")
            print(cluster_size_ratios[-1])

            steps = int(exp_max_rate / exp_removal_rate)
            title = "Random Attack - Poisson mu=%.2f " % (mu)
            plot_metric_distribution([min_path, max_path],
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title)

            plot_clustering_distribution(cluster_size_ratios,
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="clusterings",
                                     labels=[r'$S$', r'$\langle s \rangle$'],
                                     title=title)
