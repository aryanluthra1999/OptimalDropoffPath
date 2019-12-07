import networkx
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
import matplotlib.pyplot as plt


class ChristofideApproxSolver:
    def __init__(self, adj_matrix, homes_arr, soda_loc, locs):
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]
        mapping = dict(zip(self.graph, locs))
        self.netx_graph = netx.relabel_nodes(self.graph, mapping)
        self.distanceMemo = dict()
        self.start_loc = soda_loc
        self.homes = homes_arr
        homes_to_visit = self.homes.copy()
        homes_to_visit.append(self.start_loc)
        self.steiner_tree = steiner_tree(self.netx_graph, homes_to_visit, weight='weight')

    def getOddVerticies(self):
        steiner_tree = self.steiner_tree
        odd_verticies = [x for x in steiner_tree.nodes() if steiner_tree.degree(x) % 2 == 1]
        print("odd degree nodes ", odd_verticies)
        plt.figure(1)
        networkx.draw_networkx(steiner_tree, with_labels=True)
        plt.figure(2)
        networkx.draw_networkx(self.netx_graph, with_labels=True)
        plt.show()
        return odd_verticies

    def getMaximalMatching(self):
        odd_verticies = self.getOddVerticies()
        matching_set = networkx.algorithms.bipartite.basic.sets(self.netx_graph)
        print(matching_set)
        matching = networkx.algorithms.bipartite.

