import numpy as np
import utils
import networkx as netx
import student_utils as util170
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.approximation.steinertree import steiner_tree
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from pprint import pprint
from multiprocessing import Process, Manager
from itertools import chain, combinations

class OrderApproximator:
    def __init__(self, adj_matrix, homes_arr, soda_loc, locs):
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]  ## maybe we want a graph object from network x instead
        mapping = dict(zip(self.graph,locs))
        self.graph = netx.relabel_nodes(self.graph, mapping)
        self.distanceMemo = dict()
        self.start_loc = soda_loc
        self.homes = homes_arr

    def get_steiner_tree(self):
        homes_to_visit = self.homes.copy()
        homes_to_visit.append(self.start_loc)
        result = steiner_tree(self.graph, homes_to_visit, weight='weight')
        return result

    def get_dropoff_ordering(self):
        mst = self.get_steiner_tree()
        preorder_nodes = dfs_preorder_nodes(mst, source=self.start_loc)

        preorder_nodes = list(preorder_nodes)

        final_order = [n for n in preorder_nodes if n in self.homes]

        print(final_order)

        return final_order

    def get_drop_path(self):

        result = []
        curr_loc = self.start_loc

        order = self.get_dropoff_ordering()

        for i in range(len(order)):
            home = order[i]

            shortest_path = netx.shortest_path(self.graph, source=self.start_loc, target=home)

            for node in shortest_path:
                break


        return order







