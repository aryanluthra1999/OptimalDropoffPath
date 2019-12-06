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

class OrderApproximator:

    def __init__(self, adj_matrix, homes_arr, soda_loc, locs):
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]  ## maybe we want a graph object from network x instead
        mapping = dict(zip(self.graph,locs))
        self.graph = netx.relabel_nodes(self.graph, mapping)
        self.distanceMemo = dict()
        self.start_loc = soda_loc
        self.homes = homes_arr
        self.locs=locs



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


        #print(locs)
        #print("Drop-offs", dropoffs)

        return [locs,dropoffs]

    def get_steiner_tree(self):
        homes_to_visit = self.homes.copy()
        homes_to_visit.append(self.start_loc)
        result = steiner_tree(self.graph, homes_to_visit, weight='weight')
        return result


    def get_mst(self):
        return netx.minimum_spanning_tree(self.graph)


    def get_dropoff_ordering_steiner(self):
        mst = self.get_steiner_tree()
        preorder_nodes = dfs_preorder_nodes(mst, source=self.start_loc)

        preorder_nodes = list(preorder_nodes)

        final_order = [n for n in preorder_nodes if n in self.homes]

        #print(final_order)

        return final_order

    def get_dropoff_ordering_mst(self):

        mst = self.get_mst()

        preorder_nodes = dfs_preorder_nodes(mst, source=self.start_loc)

        preorder_nodes = list(preorder_nodes)

        final_order = [n for n in preorder_nodes if n in self.homes]

        #print(final_order)

        return final_order


    def get_drop_path(self, tree_func = get_dropoff_ordering_steiner):



        curr_loc = self.start_loc
        result = [curr_loc]
        #pdb.set_trace()
        order = tree_func(self)

        #print("Dropoff ordering: ", order)

        order.append(self.start_loc)

        for i in range(len(order) - 1):
            home = order[i]
            next_home = order[i+1]

            shortest_path = netx.shortest_path(self.graph, source=curr_loc, target=home)

            for node in shortest_path:
                curr_loc = node

                cont_path_cost = 2/3*(netx.dijkstra_path_length(self.graph, source=curr_loc, target=home)
                                      + netx.dijkstra_path_length(self.graph, source=home, target=next_home))

                drop_path_cost = (netx.dijkstra_path_length(self.graph, source=curr_loc, target=home)
                                  + 2/3*(netx.dijkstra_path_length(self.graph, source=curr_loc, target=next_home)))

                if drop_path_cost <= cont_path_cost:
                    result.append("drop off " + str(home))
                    break
                else:
                    result.append(node)

        result.append(curr_loc)
        result.append("go_home")

        result = self.postprocess(result)

        #print(result)
        return self.get_path_dropoffs(result)

    def postprocess(self, action_list):

        dropoff_stack = utils.Stack()
        result = []

        for action in action_list:

            if "drop off" in action:
                dropoff_stack.push(action)

            else:

                result.append(action)

                while not dropoff_stack.isEmpty():
                    result.append(dropoff_stack.pop())

        return result



    def get_drop_path_with_order(self, order):

        curr_loc = self.start_loc
        result = [curr_loc]
        #pdb.set_trace()

        #print("Dropoff ordering: ", order)

        order.append(self.start_loc)

        for i in range(len(order) - 1):
            home = order[i]
            next_home = order[i+1]

            shortest_path = netx.shortest_path(self.graph, source=curr_loc, target=home)

            for node in shortest_path:
                curr_loc = node

                cont_path_cost = 2/3*(netx.dijkstra_path_length(self.graph, source=curr_loc, target=home)
                                      + netx.dijkstra_path_length(self.graph, source=home, target=next_home))

                drop_path_cost = (netx.dijkstra_path_length(self.graph, source=curr_loc, target=home)
                                  + 2/3*(netx.dijkstra_path_length(self.graph, source=curr_loc, target=next_home)))

                if drop_path_cost <= cont_path_cost:
                    result.append("drop off " + str(home))
                    break
                else:
                    result.append(node)

        result.append(curr_loc)
        result.append("go_home")

        result = self.postprocess(result)

        #print(result)
        return self.get_path_dropoffs(result)


    """def steiner_aneal(self, iterations = 1000, tree_func = get_dropoff_ordering_steiner):

        curr_order = steiner_

        curr_result =


    def mst_aneal(self, iterations=1000, tree_func=get_dropoff_ordering_steiner):"""

