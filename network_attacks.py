import sys
import copy
import random

from tqdm import tqdm
import networkx as nx
from networkx.algorithms.shortest_paths.generic import average_shortest_path_length

from scipy.stats import mstats
import numpy as np


def get_attack_metrics(net):
    """
    Function calculating the metrics to analyze, diameter (avg and max of eccentricity)
    :param net: Attacked network
    :return: A dictionary with:
        - "shortest_paths": Network shortest path lengths between nodes.
        - "eccentricities": Network largest path lengths between nodes.
        - "cluster_sizes": Size ratios of the island/clusters/sub-nets developed after attack.
    """
    results = {}
    # Get network nodes list
    nodes = list(net.nodes)

    # Calculate shortest paths from all nodes to all other nodes
    shortest_paths = nx.shortest_path_length(net)
    shortest_path_list = []
    for node, paths in shortest_paths:
        shortest_path_list.extend(list(paths.values()))
    # Remove paths with length 0 as they are irrelevant
    shortest_path_list = np.array(shortest_path_list, dtype=np.int32)
    shortest_path_list = shortest_path_list[shortest_path_list > 0]

    # Calculate max path length between nodes and the rest of the graph
    min_cluster = None
    if nx.is_connected(net):
        nodes_eccentricity = np.array(list(nx.eccentricity(net).values()), dtype=np.int32)
        avg_size_isolated_clusters = 1.0
        relative_size_largest_cluster = 1.0
        cluster_quantiles = mstats.mquantiles([0.0, 0.0, 0.0], axis=0)
    else:
        # Find the separated insels network nodes
        components = sorted(nx.connected_components(net), key=len, reverse=True)
        cluster_sizes = [len(c) / len(nodes) for c in components]
        avg_size_isolated_clusters = np.mean([len(x) for x in components[1:]])
        relative_size_largest_cluster = cluster_sizes[0]
        # Consider eccentricitie of largest
        largest_component = components[0]
        largest_subgraph = net.subgraph(largest_component)
        nodes_eccentricity = np.array(list(nx.eccentricity(largest_subgraph).values()), dtype=np.int32)

        cluster_quantiles = mstats.mquantiles(cluster_sizes[1:], axis=0)
        # Replace 50% percentile with mean value
        cluster_quantiles[1] = np.mean(cluster_sizes[1:])

    results["shortest_paths"] = shortest_path_list
    results["eccentricities"] = nodes_eccentricity

    min_cluster = np.vstack([min_cluster, cluster_quantiles]) if min_cluster is not None else cluster_quantiles

    # Save a tuple of (S, <s>) which is the relative size of the largest cluster and the average size of the isolated clusters
    results["cluster_sizes"] = [(relative_size_largest_cluster, avg_size_isolated_clusters), min_cluster]

    return results


def incremental_random_failure(net, removal_rate, max_rate=0.5, verbose=False, track_net_num=None):
    """
    This type of attack simulates a sequential and time increasing attack, where at each iteration a constant ratio
    of the original net nodes are removed.
    :param net: Network to attack
    :param removal_rate: Ratio of the original nodes to remove at each attack step
    :param max_rate: Max ratio of removed nodes
    :return: - shortest path [25% quantile, mean, 75% quantile]
             - shortest path [25% quantile, mean, 75% quantile]
             - cluster_sizes_ratios: Relative size of the clusters/islands of networks generated after the attack.
    """

    attacked_net = net
    original_net_size = len(net.nodes)
    min_path, max_path, cluster_size_ratios = None, None, []

    steps = int(max_rate / removal_rate)
    if track_net_num is not None:
        track_net_num = int(steps/track_net_num)
        network_tracking = {0: nx.Graph(net)}
    step = 0
    for ratio_removed in tqdm([removal_rate] * steps, desc="- Incremental_failure", disable=not verbose, file=sys.stdout):
        step = step+1
        # Get list of nodes
        nodes = list(attacked_net.nodes)
        # Randomly select nodes to remove
        removed_nodes = random.sample(nodes, int(original_net_size * ratio_removed))
        # Remove the nodes
        attacked_net.remove_nodes_from(removed_nodes)

        metrics = get_attack_metrics(attacked_net)

        # min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        # max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)
        #
        # # Replace 50% percentile with mean value
        # min_quantiles[1] = np.mean(metrics["shortest_paths"])
        # max_quantiles[1] = np.mean(metrics["eccentricities"])

        min_quantiles = np.zeros(3, dtype=np.float)
        max_quantiles = np.zeros(3, dtype=np.float)

        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        std = np.std(metrics["shortest_paths"], axis=0)
        min_quantiles[2] = min_quantiles[1] + std
        min_quantiles[0] = min_quantiles[1] - std

        max_quantiles[1] = np.mean(metrics["eccentricities"])
        std = np.std(metrics["eccentricities"], axis=0)
        max_quantiles[2] = max_quantiles[1] + std
        max_quantiles[0] = max_quantiles[1] - std


        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

        if track_net_num is not None:
            if step % track_net_num == 0:
                network_tracking.update({removal_rate * step: nx.Graph(attacked_net)})

    if track_net_num is not None:
        return min_path, max_path, cluster_size_ratios, network_tracking
    return min_path, max_path, cluster_size_ratios


def instantaneous_random_failure(net, removal_rates, verbose=False):
    """
    This type of attack simulates an instantaneous attack where at each step the original network losses a randomly
    selected portion of its nodes. Each step is independent from the other.
    :param net: Network to attack
    :param removal_rates: Vector of removal rates to test
    :return: - shortest path [25% quantile, mean, 75% quantile]
             - shortest path [25% quantile, mean, 75% quantile]
             - cluster_sizes_ratios: Relative size of the clusters/islands of networks generated after the attack.
    """

    original_net_size = len(net.nodes)
    min_path, max_path, cluster_size_ratios = None, None, []

    for ratio_removed in tqdm(removal_rates, desc="- Instantaneous_attack", disable=not verbose, file=sys.stdout):
        # Create deep copy of the original network
        attacked_net = copy.deepcopy(net)
        # Get list of nodes
        nodes = list(attacked_net.nodes)
        # Randomly select nodes to remove
        removed_nodes = random.sample(nodes, int(original_net_size * ratio_removed))
        # Remove the nodes
        attacked_net.remove_nodes_from(removed_nodes)

        metrics = get_attack_metrics(attacked_net)

        # min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        # max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)
        #
        # # Replace 50% percentile with mean value
        # min_quantiles[1] = np.mean(metrics["shortest_paths"])
        # max_quantiles[1] = np.mean(metrics["eccentricities"])

        min_quantiles = np.zeros(3, dtype=np.float)
        max_quantiles = np.zeros(3, dtype=np.float)

        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        std = np.std(metrics["shortest_paths"], axis=0)
        min_quantiles[2] = min_quantiles[1] + std
        min_quantiles[0] = min_quantiles[1] - std

        max_quantiles[1] = np.mean(metrics["eccentricities"])
        std = np.std(metrics["eccentricities"], axis=0)
        max_quantiles[2] = max_quantiles[1] + std
        max_quantiles[0] = max_quantiles[1] - std


        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

    return min_path, max_path, cluster_size_ratios


def instantaneous_attack(net, removal_rates, verbose=False):
    """
    This type of attack simulates an instantaneous attack where at each step first remove the most connected node, and then
    selecting and removing nodes in decreasing order of their connectivity k.
    :param net: Network to attack
    :param removal_rates: Vector of removal rates to test
    :return: - shortest path [25% quantile, mean, 75% quantile]
             - shortest path [25% quantile, mean, 75% quantile]
             - cluster_sizes_ratios: Relative size of the clusters/islands of networks generated after the attack.
    """

    original_net_size = len(net.nodes)
    min_path, max_path, cluster_size_ratios = None, None, []

    for ratio_removed in tqdm(removal_rates, desc="- Instantaneous_attack", disable=not verbose, file=sys.stdout):
        # Create deep copy of the original network
        attacked_net = copy.deepcopy(net)
        # Get list of nodes
        nodes = list(attacked_net.nodes)


        # degree_centrality
        degree_centralities = nx.degree_centrality(attacked_net)
        degree_centralities = sorted(degree_centralities.items(), key=lambda item: item[1], reverse=True)

        # # Node connectivity is equal to the minimum number of nodes that must be removed to disconnect the node
        # connectivities = [(node, nx.degree(attacked_net, node)) for node in nodes]
        # connectivities.sort(key=lambda x: x[1], reverse=True)

        num_removed_nodes = int(original_net_size * ratio_removed)
        removed_nodes = [x[0] for x in degree_centralities[:num_removed_nodes]]

        # Remove the nodes
        attacked_net.remove_nodes_from(removed_nodes)

        metrics = get_attack_metrics(attacked_net)

        # min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        # max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)
        #
        # # Replace 50% percentile with mean value
        # min_quantiles[1] = np.mean(metrics["shortest_paths"])
        # max_quantiles[1] = np.mean(metrics["eccentricities"])

        min_quantiles = np.zeros(3, dtype=np.float)
        max_quantiles = np.zeros(3, dtype=np.float)

        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        std = np.std(metrics["shortest_paths"], axis=0)
        min_quantiles[2] = min_quantiles[1] + std
        min_quantiles[0] = min_quantiles[1] - std

        max_quantiles[1] = np.mean(metrics["eccentricities"])
        std = np.std(metrics["eccentricities"], axis=0)
        max_quantiles[2] = max_quantiles[1] + std
        max_quantiles[0] = max_quantiles[1] - std


        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

    return min_path, max_path, cluster_size_ratios


def incremental_attack(net, removal_rate, max_rate=0.5, verbose=False, track_net_num=None):
    """
    This type of attack simulates an instantaneous attack where at each step first remove the most connected node, and then
    selecting and removing nodes in decreasing order of their connectivity k.
    :param net: Network to attack
    :param removal_rates: Vector of removal rates to test
    :param track_net_num: Number of steps to store the current network (approximately)
    :return: - shortest path [25% quantile, mean, 75% quantile]
             - shortest path [25% quantile, mean, 75% quantile]
             - cluster_sizes_ratios: Relative size of the clusters/islands of networks generated after the attack.
    """

    attacked_net = nx.Graph(net)
    original_net_size = len(net.nodes)
    min_path, max_path, cluster_size_ratios = None, None, []

    steps = int(max_rate / removal_rate)
    if track_net_num is not None:
        track_net_num = int(steps/track_net_num)
        network_tracking = {0: nx.Graph(net)}
    step = 0
    for ratio_removed in tqdm([removal_rate] * steps, desc="- Incremental_attack", disable=not verbose, file=sys.stdout):
        step = step+1
        # Create deep copy of the original network
        # Get list of nodes
        nodes = list(attacked_net.nodes)

        # degree_centrality
        degree_centralities = nx.degree_centrality(attacked_net)
        degree_centralities = sorted(degree_centralities.items(), key=lambda item: item[1], reverse=True)

        # # Node connectivity is equal to the minimum number of nodes that must be removed to disconnect the node
        # connectivities = [(node, nx.degree(attacked_net, node)) for node in nodes]
        # connectivities.sort(key=lambda x: x[1], reverse=True)

        num_removed_nodes = int(original_net_size * ratio_removed)
        removed_nodes = [x[0] for x in degree_centralities[:num_removed_nodes]]

        # Remove the nodes
        attacked_net.remove_nodes_from(removed_nodes)

        metrics = get_attack_metrics(attacked_net)

        # min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        # max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)
        # Replace 50% percentile with mean value
        # min_quantiles[1] = np.mean(metrics["shortest_paths"])
        # max_quantiles[1] = np.mean(metrics["eccentricities"])


        min_quantiles = np.zeros(3, dtype=np.float)
        max_quantiles = np.zeros(3, dtype=np.float)

        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        std = np.std(metrics["shortest_paths"], axis=0)
        min_quantiles[2] = min_quantiles[1] + std
        min_quantiles[0] = min_quantiles[1] - std

        max_quantiles[1] = np.mean(metrics["eccentricities"])
        std = np.std(metrics["eccentricities"], axis=0)
        max_quantiles[2] = max_quantiles[1] + std
        max_quantiles[0] = max_quantiles[1] - std


        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

        if track_net_num is not None:
            if step % track_net_num == 0:
                network_tracking.update({removal_rate * step: nx.Graph(attacked_net)})

    if track_net_num is not None:
        return min_path, max_path, cluster_size_ratios, network_tracking
    return min_path, max_path, cluster_size_ratios



