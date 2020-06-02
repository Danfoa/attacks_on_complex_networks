import os 
import numpy as np
import pickle as pkl

def save_results(min_path, max_path, cluster_size_ratios, file_name):

    os.makedirs(file_name, exist_ok=True)

    min_path_file = os.path.join(file_name, "min_path.pickle")
    max_path_file = os.path.join(file_name, "max_path.pickle")
    clusterings_file = os.path.join(file_name, "cluster_size_ratios.pickle")

    fileObject = open(min_path_file, 'wb')
    pkl.dump(min_path, fileObject)
    fileObject.close()

    fileObject = open(max_path_file, 'wb')
    pkl.dump(max_path, fileObject)
    fileObject.close()

    fileObject = open(clusterings_file, 'wb')
    pkl.dump(cluster_size_ratios, fileObject)
    fileObject.close()



def load_results(file_name):
    min_path, max_path, cluster_size_ratios = None, None, None 
    if os.path.exists(file_name):
        min_path_file = os.path.join(file_name, "min_path.pickle")
        max_path_file = os.path.join(file_name, "max_path.pickle")
        clusterings_file = os.path.join(file_name, "cluster_size_ratios.pickle")

        fileObject = open(min_path_file, 'rb')
        min_path = pkl.load(fileObject)
        fileObject.close()

        fileObject = open(max_path_file, 'rb')
        max_path = pkl.load(fileObject)
        fileObject.close()

        fileObject = open(clusterings_file, 'rb')
        cluster_size_ratios = pkl.load(fileObject)
        fileObject.close()

        print(min_path)

    return min_path, max_path, cluster_size_ratios



