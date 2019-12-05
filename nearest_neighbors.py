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
    def get_path_dropoffs(self, result):
        locs=[]
        dropoffs=dict()
        curr=''
        for i in range(len(result)):
            if curr=='':
                curr=result[i]
                locs.append(curr)
            elif result[i]==curr:
                continue
            elif len(result[i])>8 and result[i][0:8]=='drop off':
                if curr not in dropoffs.keys():
                    dropoffs[curr]=[result[i][9:]]
                else:
                    dropoffs[curr]=dropoffs[curr]+[result[i][9:]]
            elif len(result[i])==7 and result[i]=='go_home':
                break
            else:
                curr=result[i]
                locs.append(curr)
        back_home=netx.shortest_path(self.graph,locs[len(locs)-1],locs[0])
        #print(back_home)
        for i in range(len(back_home)):
            if i==0:
                continue
            else:
                locs.append(back_home[i])
        return [locs,dropoffs]

    def get_dropoff_ordering_ns(self):
        homes_visited=[False for i in self.homes]
        loc=self.start_loc
        home_dists=dict()
        path=[self.start_loc]
        index_dict=dict()
        while len([i for i in homes_visited if i==False])!=0:
            for i in range(len(self.homes)):
                if homes_visited[i]==False:
                    index_dict[self.homes[i]]=i
                    print(self.homes[i])
                    print(netx.dijkstra_path_length(self.graph, source=loc, target=self.homes[i]),'\n')
                    home_dists[self.homes[i]]=netx.dijkstra_path_length(self.graph, source=loc, target=self.homes[i])
            home_to_visit=min(home_dists, key=home_dists.get)
            homes_visited[index_dict[home_to_visit]]=True
            path_to_home=netx.shortest_path(self.graph, source=loc, target=home_to_visit)
            for node in path_to_home:
                path.append(node)
            path.append('drop off '+home_to_visit)
            loc=home_to_visit
            index_dict=dict()
            home_dists=dict()
        path.append('go_home')
        return self.get_path_dropoffs(path)
