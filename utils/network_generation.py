import time
import warnings

import networkx as nx
import numpy as np
import powerlaw
from scipy.stats import poisson

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


def get_gnutella():
    filename = "networks/p2p-Gnutella08.txt"
    G = nx.read_edgelist(filename)
    return G