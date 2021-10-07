# Attacks on Complex Networks

This repository holds the source code and the final report of the Complex Networks course from [Prof. Dr. Alex Arenas] of the Artificial Intelligence Master at UPC and URV.

In  this  work,  we examine the resiliance of two complex network types (Erdos  Renyi, and Power-Law/Scale free) to potential delivered attacks and random errors.  

With respect to targeted attacks, in which the nodes with highest centrality are removed, the experimental results show that scale-free networks are more susceptible,  because in these architectures the nodes with higher centrality are the nodes with higher degree, resulting in a fast fragmentation of the network. On the other hand, Poisson networks seem to be more robust to targeted attacks, presenting less fragmentation of the network and greater stability of the main network connectivity. 

With respect to failures, in which pseudo-randomly selected nodes are removed from the network, scale-free networks present great robustness, because the vast majority of its nodes are not relevant for the overall network connectivity. On the contrary, Poisson networks result more susceptible to failures, susceptibility that decreases with the increase of the mean degree of the nodes in the network. A behaviour that might be explained by the fact that a high average degree implies the existence of multiple redundant  paths between all of the nodes in the network.

For further details see the Project Report, and the `experiments.py` script for reproducing the results. 
_______________________________________
![image](https://user-images.githubusercontent.com/8356912/136397939-4a7540f0-89cb-4c5c-8d05-56aa524bbdd4.png)
_______________________________________
![image](https://user-images.githubusercontent.com/8356912/136398061-124767c7-fa4d-4644-bcf7-cf6b92e1f27c.png)
