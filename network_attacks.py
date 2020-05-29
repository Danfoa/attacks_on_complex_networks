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
    if nx.is_connected(net):
        nodes_eccentricity = np.array(list(nx.eccentricity(net).values()), dtype=np.int32)
        cluster_sizes = [1.0]
    else:
        # Find the separated insels network nodes
        components = sorted(nx.connected_components(net), key=len, reverse=True)
        cluster_sizes = [len(c) / len(nodes) for c in components]
        # Consider eccentricitie of largest
        largest_component = components[0]
        largest_subgraph = net.subgraph(largest_component)
        nodes_eccentricity = np.array(list(nx.eccentricity(largest_subgraph).values()), dtype=np.int32)

    results["shortest_paths"] = shortest_path_list
    results["eccentricities"] = nodes_eccentricity
    results["cluster_sizes"] = cluster_sizes

    return results


def incremental_random_attack(net, removal_rate, max_rate=0.5, verbose=False):
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
    for ratio_removed in tqdm([removal_rate] * steps, desc="- Incremental_attack", disable=not verbose, file=sys.stdout):
        # Get list of nodes
        nodes = list(attacked_net.nodes)
        # Randomly select nodes to remove
        removed_nodes = random.sample(nodes, int(original_net_size * ratio_removed))
        # Remove the nodes
        attacked_net.remove_nodes_from(removed_nodes)

        metrics = get_attack_metrics(attacked_net)

        min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)

        # Replace 50% percentile with mean value
        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        max_quantiles[1] = np.mean(metrics["eccentricities"])

        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

    return min_path, max_path, cluster_size_ratios


def instantaneous_random_attack(net, removal_rates, verbose=False):
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

        min_quantiles = mstats.mquantiles(metrics["shortest_paths"], axis=0)
        max_quantiles = mstats.mquantiles(metrics["eccentricities"], axis=0)

        # Replace 50% percentile with mean value
        min_quantiles[1] = np.mean(metrics["shortest_paths"])
        max_quantiles[1] = np.mean(metrics["eccentricities"])

        min_path = np.vstack([min_path, min_quantiles]) if min_path is not None else min_quantiles
        max_path = np.vstack([max_path, max_quantiles]) if max_path is not None else max_quantiles
        cluster_size_ratios.append(metrics["cluster_sizes"])

    return min_path, max_path, cluster_size_ratios




