import os 
import numpy as np

def save_results(min_path, max_path, cluster_size_ratios, file_name):

    os.makedirs(file_name, exist_ok=True)

    np.save(os.path.join(file_name, "min_path.npz"), min_path)
    np.save(os.path.join(file_name, "max_path.npz"), max_path)
    np.save(os.path.join(file_name, "cluster_size_ratios.npz"), cluster_size_ratios)


def load_results(file_name):
    min_path, max_path, cluster_size_ratios = None, None, None 
    if os.path.exists(file_name):
        min_path = np.load(os.path.join(file_name, "min_path.npz"))
        max_path = np.load(os.path.join(file_name, "max_path.npz"))
        cluster_size_ratios = np.load(os.path.join(file_name, "cluster_size_ratios.npz"))

    return min_path, max_path, cluster_size_ratios



