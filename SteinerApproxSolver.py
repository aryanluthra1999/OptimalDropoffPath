#  Use Steiner's Approximation to solver DTH
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



class SteinerApproxSolver:
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


    # def getSteinerTree(self):
    #     homes_to_visit = self.homes.copy()
    #     homes_to_visit.append(self.start_loc)
    #     self.steiner_tree = steiner_tree(self.netx_graph, homes_to_visit, weight='weight')

    def getOrdering(self):
        steiner_tree = self.steiner_tree
        preorder_nodes = dfs_preorder_nodes(steiner_tree, source=self.start_loc)
        final_order = [n for n in preorder_nodes if n in self.homes]
        return final_order

    def getLeafNodes(self):
        steiner_tree = self.steiner_tree
        leaf_homes = [x for x in steiner_tree.nodes() if steiner_tree.degree(x) == 1 and x != self.start_loc]
        print("leaf nodes ", leaf_homes)
        plt.figure(1)
        networkx.draw_networkx(steiner_tree,with_labels=True)
        # plt.figure(2)
        # networkx.draw_networkx(self.netx_graph, with_labels=True)
        plt.show()
        return leaf_homes

    def solveSteinerTreeDTH(self):
        print("start node: ", self.start_loc)
        print("home nodes: ", self.homes)
        # print("Traversal actual with homes: ", traversal_ordering)
        leaf_homes = self.getLeafNodes()
        preorder_nodes = dfs_preorder_nodes(self.steiner_tree, source=self.start_loc)
        traversal_ordering = [n for n in preorder_nodes if n in leaf_homes]
        print("Traversal ordering of the leaf: ",traversal_ordering)
        # """Remove non-leaf nodes from the order."""
        # for home in traversal_ordering:
        #     if home not in leaf_homes:
        #         print(home)
        #         traversal_ordering.remove(home)
        # current_loc = self.start_loc
        # """ Needs to start and end at root"""
        # print("Traversal_ordering: ", traversal_ordering)
        # # traversal_ordering.insert(0, current_loc)
        # # traversal_ordering.append(current_loc)

        """Hash map of the dropoffs"""
        dropoffs = dict()
        for i in range(len(traversal_ordering) - 1):
            current_leaf_home = traversal_ordering[i]
            next_leaf_home = traversal_ordering[i+1]
            """Shortest path between current and next leaf home on the graph"""
            shortest_path = netx.shortest_path(self.netx_graph, source=current_leaf_home, target=next_leaf_home)
            print("Shortest path between ", current_leaf_home, " and ", next_leaf_home, shortest_path)

            for node in shortest_path:
                """Check if any node in the shortest node is a part of the Steiner Tree """
                if node != current_leaf_home and node != next_leaf_home and node in self.steiner_tree:
                    #print("Node : ", node)
                    """Shares a common ancestor."""
                    """Drop off curr_node at curr->parent"""
                    #print("Steiner tree: ", set(self.steiner_tree))
                    print("Dfs ",networkx.dfs_predecessors(self.steiner_tree, source=self.start_loc))
                    dropoffs[current_leaf_home] = networkx.dfs_predecessors(self.steiner_tree, source=self.start_loc)[current_leaf_home]
        print("Dropoffs ", dropoffs)

        new_candidate_dropoff = set()
        for home in self.homes:
            if home in dropoffs.keys():
                new_candidate_dropoff.add(dropoffs[home])
            else:
                new_candidate_dropoff.add(home)
        print("homes ", self.homes)
        new_candidate_dropoff = list(new_candidate_dropoff)


        """Add source to the candidate dropoff list to create the steiner tree."""
        new_candidate_dropoff.append(self.start_loc)
        #print("Candidate_dropoffs ", new_candidate_dropoff)
        steiner_tree_candidate_dropoffs = steiner_tree(self.netx_graph, new_candidate_dropoff, weight='weight')
        preorder_nodes = dfs_preorder_nodes(steiner_tree_candidate_dropoffs, source=self.start_loc)
        preorder_nodes = list(preorder_nodes)
        final_order = [n for n in preorder_nodes if n in steiner_tree_candidate_dropoffs]
        #print("final order pre", final_order)

        for elem in final_order:
            if elem in dropoffs.values():
                keys = [k for k,v in dropoffs.items() if v == elem]
                if elem in self.homes or elem == self.start_loc:
                    for i in keys:
                        final_order.insert(final_order.index(elem)+1, elem+" "+i)
                else:
                    index = final_order.index(elem)
                    for i in keys:
                        final_order[index] = elem+" "+i
                        index = index+1
        print("final order ", final_order)
        return self.get_cost_params(final_order)

    def get_cost_params(self,result):
        locs=[]
        dropoffs=dict()
        curr=''
        for i in range(len(result)):
            if curr=='':
                curr=result[i]
                if curr in self.homes and curr not in dropoffs.keys():
                    dropoffs[curr]=[result[i][9:]]
                elif curr in self.homes and curr in dropoffs.keys():
                    dropoffs[curr]+=[result[i][9:]]
                locs.append(curr)
            elif ' ' in result[i]:
                curr=result[i]
                if curr[0:curr.index(' ')] not in dropoffs.keys():
                    dropoffs[curr[0:curr.index(' ')]]=[result[i][result[i].index(' ')+1:]]
                    locs.append(curr[0:curr.index(' ')])
                else:
                    dropoffs[curr[0:curr.index(' ')]]=dropoffs[curr[0:curr.index(' ')]]+[result[i][result[i].index(' ')+1:]]
            elif i==len(result)-1:
                locs.append(result[i])
                if result[i] in self.homes:
                    dropoffs[result[i]]=[result[i]]
                break
            else:
                curr=result[i]
                locs.append(curr)
                if curr in self.homes:
                    dropoffs[curr]=[result[i]]
        back_home=netx.shortest_path(self.netx_graph,locs[len(locs)-1],locs[0])
        #print(back_home)
        for i in range(len(back_home)):
            if i==0:
                continue
            else:
                locs.append(back_home[i])
        print(locs)
        print(dropoffs)
        return locs,dropoffs
