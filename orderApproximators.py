import numpy as np
import utils
import networkx as netx
import student_utils as util170
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.approximation.steinertree import steiner_tree
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


    def get_dropoff_ordering(self):
        homes_to_visit = []
        homes_to_visit.append(self.start_loc)
