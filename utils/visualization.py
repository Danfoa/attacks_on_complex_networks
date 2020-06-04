import time
import warnings

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
import os
from scipy.stats import poisson
import networkx as nx

from network_attacks import instantaneous_attack, incremental_attack, incremental_random_failure, instantaneous_random_failure
from utils.configuration_model import ConfigurationGenerator
from utils.data_saver import save_results, load_results
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
    filename = os.path.join('..', 'results', filename)
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
    filename = os.path.join('..', 'results', filename)
    plt.savefig(filename)


def plot_network_tracking(network_tracking, title, file_name):
    ratios = sorted(list(network_tracking.keys()))
    num_plots = len(ratios)
    cols = int((num_plots + 1) / 2)
    size = 5
    plt.figure(figsize=(cols * size, 2 * size))
    pos = nx.random_layout(network_tracking[ratios[0]])
    for i, ratio in enumerate(ratios):
        filename = os.path.join(file_name, "network_tracking({}).net".format(i))
        nx.write_pajek(network_tracking[ratio], filename)
        plt.subplot(2, cols, i + 1)
        nx.draw(network_tracking[ratio], node_size=10, pos=pos)
        plt.title("Ratio:{} - N:{}".format(ratio, network_tracking[ratio].number_of_nodes()))

    plt.savefig("../results/{}_network_tracking.png".format(file_name))


def incremental_attack_poisson(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_mus, is_random_attack, track_net=1):
    for n_nodes in exp_num_nodes:
        for mu in exp_mus:
            net = get_poisson_net(n_nodes=n_nodes, mu=mu, verbose=False)
            if not is_random_attack:
                file_name = "poisson-incr-attack-mu=%.3f-n_nodes=%d" % (mu, n_nodes)
                title = "Incremental Attack - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)

                min_path, max_path, cluster_size_ratios, network_tracking = incremental_attack(net=net,
                                                                                               removal_rate=exp_removal_rate,
                                                                                               max_rate=exp_max_rate,
                                                                                               verbose=True,
                                                                                               track_net_num=track_net)

            else:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_random_failure(net=net,
                                                                                                       removal_rate=exp_removal_rate,
                                                                                                       max_rate=exp_max_rate,
                                                                                                       track_net_num=track_net)

                file_name = "poisson-incr-failure-mu=%.3f-n_nodes=%d" % (mu, n_nodes)
                title = "Incremental Failure - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)
            save_results(min_path, max_path, cluster_size_ratios, file_name)
            plot_network_tracking(network_tracking, title, file_name)
            steps = int(exp_max_rate / exp_removal_rate)
            plot_metric_distribution([min_path, max_path],
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')

            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         np.cumsum([exp_removal_rate] * steps),
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def instantaneous_attack_poisson(exp_num_nodes, exp_removal_ratios, exp_mus, is_random_attack):
    for n_nodes in exp_num_nodes:
        for mu in exp_mus:
            net = get_poisson_net(n_nodes=n_nodes, mu=mu, verbose=True)

            if not is_random_attack:
                min_path, max_path, cluster_size_ratios = instantaneous_attack(net=net, removal_rates=exp_removal_ratios, verbose=True)

                file_name = "poisson-inst-attack-mu=%.3f-n_nodes=%d" % (mu, n_nodes)
                title = "Instantaneous Attack - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)

            else:
                min_path, max_path, cluster_size_ratios = instantaneous_random_failure(net=net,
                                                                               removal_rates=exp_removal_ratios,
                                                                               verbose=True)

                file_name = "poisson-inst-failure-mu=%.3f-n_nodes=%d" % (mu, n_nodes)
                title = "Instantaneous Failure - Poisson nodes=%d mu=%.2f " % (n_nodes, mu)
            save_results(min_path, max_path, cluster_size_ratios, file_name)

            plot_metric_distribution([min_path, max_path],
                                     exp_removal_ratios,
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')
            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         exp_removal_ratios,
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def incremental_attack_powerlaw(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_ks, is_random_attack, track_net=1):
    for n_nodes in exp_num_nodes:
        for k in exp_ks:
            net = get_power_law_net(n_nodes, k, verbose=False)
            if not is_random_attack:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_attack(net=net,
                                                                                               removal_rate=exp_removal_rate,
                                                                                               max_rate=exp_max_rate,
                                                                                               track_net_num=track_net)

                file_name = "powerlaw-incr-attack-k=%.3f-n_nodes=%d" % (k, n_nodes)
                title = "Incremental Attack - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)
            else:
                min_path, max_path, cluster_size_ratios, network_tracking = incremental_random_failure(net=net,
                                                                                                       removal_rate=exp_removal_rate,
                                                                                                       max_rate=exp_max_rate,
                                                                                                       track_net_num=track_net)

                file_name = "powerlaw-incr-failure-k=%.3f-n_nodes=%d" % (k, n_nodes)
                title = "Incremental Failure - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)

            save_results(min_path, max_path, cluster_size_ratios, file_name)
            plot_network_tracking(network_tracking, title, file_name)
            steps = int(exp_max_rate / exp_removal_rate)

            plot_metric_distribution([min_path, max_path],
                                     np.cumsum([exp_removal_rate] * steps),
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')

            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         np.cumsum([exp_removal_rate] * steps),
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')


def instantaneous_attack_powerlaw(exp_num_nodes, exp_removal_ratios, exp_ks, is_random_attack):
    for n_nodes in exp_num_nodes:
        for k in exp_ks:
            net = get_power_law_net(n_nodes, k, verbose=False)
            if not is_random_attack:
                min_path, max_path, cluster_size_ratios = instantaneous_attack(net=net, removal_rates=exp_removal_ratios,
                                                                               verbose=True)

                file_name = "powerlaw-inst-attack-k=%.3f-n_nodes=%d" % (k, n_nodes)
                title = "Instantaneous Attack - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)
            else:
                min_path, max_path, cluster_size_ratios = instantaneous_random_failure(net=net,
                                                                               removal_rates=exp_removal_ratios,
                                                                               verbose=True)

                file_name = "powerlaw-inst-failure-k=%.3f-n_nodes=%d" % (k, n_nodes)
                title = "Instantaneous Failure - Powerlaw nodes=%d k=%.2f " % (n_nodes, k)

            save_results(min_path, max_path, cluster_size_ratios, file_name)
            plot_metric_distribution([min_path, max_path],
                                     exp_removal_ratios,
                                     y_label="path length",
                                     labels=[r'$d_{min}$', r'$d_{max}$'],
                                     title=title,
                                     filename=file_name + '_distr.png')
            clusters_info = [x[0] for x in cluster_size_ratios]
            plot_clustering_distribution(clusters_info,
                                         exp_removal_ratios,
                                         y_label="clusterings",
                                         labels=[r'$S$', r'$\langle s \rangle$'],
                                         title=title,
                                         filename=file_name + '_clust.png')



if __name__ == "__main__":
    exp_removal_rate = 0.025
    exp_removal_ratios = np.linspace(0.0, 0.5, 10)

    exp_max_rate = 0.5
    # exp_num_nodes = [2000]
    exp_num_nodes = [100]  # Test the attacks with different sizes of networks
    exp_mus = [4]
    exp_ks = [2.6]

    for is_random_attack in [False]:
        # Poisson
        incremental_attack_poisson(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_mus, is_random_attack, track_net=7)
        #instantaneous_attack_poisson(exp_num_nodes, exp_removal_ratios, exp_mus, is_random_attack)

        # Scale Free
        incremental_attack_powerlaw(exp_removal_rate, exp_max_rate, exp_num_nodes, exp_ks, is_random_attack, track_net=7)
        #instantaneous_attack_powerlaw(exp_num_nodes, exp_removal_ratios, exp_ks, is_random_attack)