
import networkx as nx
import numpy as np

def get_network():
    filename = "networks/oregon2_010526.txt"
    filename = "networks/p2p-Gnutella08.txt"
    G = nx.read_edgelist(filename)
    return G


def degree_distribution(net):
    degrees = np.array([x[1] for x in net.degree()])
    counts = np.unique(degrees, return_counts=True)

