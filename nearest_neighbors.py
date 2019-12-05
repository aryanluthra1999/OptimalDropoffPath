import numpy as np
import utils
import networkx as netx
import student_utils as util170
from networkx.utils import pairwise
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from networkx.algorithms.approximation.steinertree import steiner_tree, metric_closure
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from pprint import pprint
from multiprocessing import Process, Manager
from itertools import chain, combinations
import copy
import time
#import pdb
from tqdm import tqdm

class NearestNeighbors:

    def __init__(self, adj_matrix, homes_arr, soda_loc, locs):
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]  ## maybe we want a graph object from network x instead
        mapping = dict(zip(self.graph,locs))
        self.graph = netx.relabel_nodes(self.graph, mapping)
        self.distanceMemo = dict()
        self.start_loc = soda_loc
        self.homes = homes_arr

    def get_dropoff_ordering_ns(self):
        homes_visited=[False for i in self.homes]
        loc=self.start_loc
        home_dists=dict()
        path=[self.start_loc]
        index_dict=dict()
        while len([i for in homes_visited if i==False])!=0:
            for i in range(len(self.homes)):
                if homes_visited[i]==False:
                    index_dict[sefl.homes[i]]=i
                    home_dists[self.homes[i]]=netx.dijkstra_path_length(self.graph, source=loc, target=self.homes[i])
            home_to_visit=max(home_dists, key=home_dists.get)
            homes_visited[index_dict[home_to_visit]]=True
            path_to_home=netx.shortest_path(self.graph, source=curr_loc, target=home_to_visit)
            for node in path_to_home:
                path.append(node)
            path.append('drop off '+home_to_visit)
            loc=home_to_visit
        path.append('go_home')
        return path
