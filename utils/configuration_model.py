import random
import numpy as np
import networkx as nx


class ConfigurationGenerator:
    """
    Implementation of the configuration model producing graphs without self-loops or parallel edges
    """

    def __init__(self, degrees):
        """
        :param degrees: Array of degrees of each of the nodes
        """
        self.n = len(degrees)
        self.nodes = range(self.n)
        self.degrees = degrees

        # Ensure stubs are even
        if np.sum(self.degrees) % 2 != 0:
            self.degrees[0] += 1

        # Define the network edges and a hashable map of node connections
        self.edges = []
        self.node_neighboors = {node: set() for node in self.nodes}

        # Generate array of available stubs
        self.stubs = np.array([], dtype=np.int32)
        for node in self.nodes:
            self.stubs = np.append(self.stubs, np.ones(self.degrees[node], dtype=np.int32) * node)
        # Obtain a random permutation of the stubs
        permutation_indx = random.sample(range(len(self.stubs)), len(self.stubs))
        self.stubs = self.stubs[permutation_indx]

    def add_edge(self, n1, n2):
        edge_added = False
        # If no self connection, or recurrent connection
        if n1 != n2 and n2 not in self.node_neighboors[n1]:
            self.edges.append((n1, n2))
            self.node_neighboors[n1].add(n2)
            self.node_neighboors[n2].add(n1)
            edge_added = True
        return edge_added

    def get_network(self):
        removed_stubs = 0
        while len(self.stubs) > 2:
            node_a = self.stubs[0]
            node_b = self.stubs[1]

            # If no self connection, or recurrent connection
            if self.add_edge(node_a, node_b):
                # Return a view of the remaining stubs
                self.stubs = self.stubs[2:]

            else:  # Find random stub for connection
                solved = False
                for new_stub_idx in range(2, len(self.stubs)):
                    node_c = self.stubs[new_stub_idx]
                    if self.add_edge(node_a, node_c):
                        # Swith positions of node_b's stub with node_c's stub
                        self.stubs[new_stub_idx] = node_b
                        # Return a view of the remaining stubs
                        self.stubs = self.stubs[2:]
                        solved = True
                        break
                    elif node_a != node_b and self.add_edge(node_b, node_c):
                        # Swith positions of node_a's stub with node_c's stub
                        self.stubs[new_stub_idx] = node_a
                        # Return a view of the remaining stubs
                        self.stubs = self.stubs[2:]
                        solved = True
                        break
                if not solved:  # No available stub for node_a or node_b
                    # Remove stubs TODO: Check other methodologies
                    self.stubs = self.stubs[2:]
                    self.degrees[node_a] -= 1
                    self.degrees[node_b] -= 1
                    removed_stubs += 2

        ratio_removed_stubs = 2 * removed_stubs / len(self.edges)
        if ratio_removed_stubs > 0:
            print("Removed stubs: %.5f%%" % ratio_removed_stubs)

        self.purge_islands()
        return nx.Graph(self.edges)

    def purge_islands(self):
        # Convert nodes ids to set for performance reasons (search is on avg O(1))
        non_visited_nodes = set(self.nodes)
        main_land_nodes = set()

        seed_node = non_visited_nodes.pop()  # Sample and remove on of the non-visited nodes
        while len(non_visited_nodes) > 0:
            main_land_nodes.add(seed_node)
            main_land_nodes = self.visit_island_nodes(seed_node, main_land_nodes)
            # Mark as visited all nodes in current main_land
            non_visited_nodes.difference_update(main_land_nodes)
            if len(main_land_nodes) != len(self.nodes): # There is still an island somewhere
                outsider_node = non_visited_nodes.pop()
                self.add_edge(seed_node, outsider_node)
                self.degrees[seed_node] += 1
                self.degrees[outsider_node] += 1
                seed_node = outsider_node

    def visit_island_nodes(self, node, visited_island_members):

        neighbors = [n for n in self.node_neighboors[node] if n not in visited_island_members]

        if len(neighbors) == 0:
            return visited_island_members

        visited_island_members.update(neighbors)

        for neighbor in neighbors:
            return self.visit_island_nodes(neighbor, visited_island_members)