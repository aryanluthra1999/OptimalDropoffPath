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
import random

#import pdb
from tqdm import tqdm


def cost_of_solution(G, car_cycle, dropoff_mapping, shortest):
    cost = 0
    message = ''
    dropoffs = dropoff_mapping.keys()
    if not util170.is_valid_walk(G, car_cycle):
        message += 'This is not a valid walk for the given graph.\n'
        cost = 'infinite'

    if not car_cycle[0] == car_cycle[-1]:
        message += 'The start and end vertices are not the same.\n'
        cost = 'infinite'
    if cost != 'infinite':
        if len(car_cycle) == 1:
            car_cycle = []
        else:
            car_cycle = util170.get_edges_from_path(car_cycle[:-1]) + [(car_cycle[-2], car_cycle[-1])]
        if len(car_cycle) != 1:
            driving_cost = sum([G.edges[e]['weight'] for e in car_cycle]) * 2 / 3
        else:
            driving_cost = 0
        walking_cost = 0
        #shortest = dict(netx.floyd_warshall(G))

        for drop_location in dropoffs:
            for house in dropoff_mapping[drop_location]:
                walking_cost += shortest[drop_location][house]

        message += f'The driving cost of your solution is {driving_cost}.\n'
        message += f'The walking cost of your solution is {walking_cost}.\n'
        cost = driving_cost + walking_cost

    message += f'The total cost of your solution is {cost}.\n'
    return cost, message


class OrderApproximator:

    def __init__(self, adj_matrix, homes_arr, soda_loc, locs):
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]  ## maybe we want a graph object from network x instead
        mapping = dict(zip(self.graph,locs))
        self.graph = netx.relabel_nodes(self.graph, mapping)
        self.start_loc = soda_loc
        self.homes = homes_arr
        self.locs=locs
        self.distanceMemo = dict()

        self.distancePathMemo = dict(netx.algorithms.shortest_paths.all_pairs_dijkstra_path(self.graph))
        self.distanceLengthMemo = dict(netx.algorithms.shortest_paths.all_pairs_dijkstra_path_length(self.graph))
        print("finished_memoizing_shortest_paths")



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
        back_home = self.distancePathMemo[locs[len(locs)-1]][locs[0]]
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

    def distance(self, source, target):
        k = (source, target)
        if k in self.distanceMemo:
            return self.distanceMemo[k]
        else:
            result = netx.dijkstra_path(self.graph, source, target), netx.dijkstra_path_length(self.graph, source, target)
            self.distanceMemo[k] = result
            return result

    def get_drop_path(self, tree_func = get_dropoff_ordering_steiner):



        curr_loc = self.start_loc
        result = [curr_loc]
        #pdb.set_trace()
        order = tree_func(self)

        #print("Dropoff ordering: ", order)

        order.append(self.start_loc)

        for i in range(len(order) - 1):
            home = order[i]
            next_home = order[i + 1]

            shortest_path = self.distancePathMemo[curr_loc][home]

            for node in shortest_path:
                curr_loc = node

                cont_path_cost = 2 / 3 * (self.distanceLengthMemo[curr_loc][home]
                                          + self.distanceLengthMemo[home][next_home])

                drop_path_cost = (self.distanceLengthMemo[curr_loc][home]
                                  + 2 / 3 * self.distanceLengthMemo[curr_loc][next_home])

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



    def get_drop_path_with_order(self, order_orr):

        order = order_orr.copy()

        curr_loc = self.start_loc
        result = [curr_loc]
        #pdb.set_trace()

        #print("Dropoff ordering: ", order)

        order.append(self.start_loc)

        for i in range(len(order) - 1):
            home = order[i]
            next_home = order[i + 1]

            shortest_path = self.distancePathMemo[curr_loc][home]

            for node in shortest_path:
                curr_loc = node

                cont_path_cost = 2 / 3 * (self.distanceLengthMemo[curr_loc][home]
                                          + self.distanceLengthMemo[home][next_home])

                drop_path_cost = (self.distanceLengthMemo[curr_loc][home]
                                  + 2 / 3 * self.distanceLengthMemo[curr_loc][next_home])

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


    def steiner_aneal(self, iterations = 270000, tree_func = get_dropoff_ordering_steiner, epsilon = .05):

        curr_order = tree_func(self)
        curr_result = self.get_drop_path_with_order(curr_order)
        curr_min = cost_of_solution(self.graph, curr_result[0], curr_result[1], self.distanceLengthMemo)
        # print("initial cost ", curr_min)

        i = 0

        while i <= (iterations):
            #print(i)

            e_prob = epsilon


            new_order = self.mutate_order(curr_order)
            new_result = self.get_drop_path_with_order(new_order)
            new_cost = cost_of_solution(self.graph, new_result[0], new_result[1], self.distanceLengthMemo)

            # print(new_order)

            if new_cost < curr_min or random.random() < e_prob:
                curr_order = new_order
                curr_result = new_result
                curr_min = new_cost

                i = 0

            i += 1


        return curr_result

    def mutate_order(self, order):

        arr = copy.deepcopy(order)
        # print("After deep copy :", len(arr))

        if len(arr) <=1:
            return arr
        else:
            pos1 = int(random.random()*len(arr))
            pos2 = int(random.random()*len(arr))
            arr[pos1], arr[pos2] = arr[pos2], arr[pos1]
            # print("In else block: ", len(arr))
            return arr


    def mst_aneal(self, iterations=100, tree_func=get_dropoff_ordering_mst, epsilon = 0.7):
        curr_order = tree_func(self)
        curr_result = self.get_drop_path_with_order(curr_order)
        curr_min = cost_of_solution(self.graph, curr_result[0], curr_result[1], self.distanceLengthMemo)
        # print("initial cost ", curr_min)

        for i in (range(iterations)):

            e_prob = epsilon / max(1, i / 10)

            new_order = self.mutate_order(curr_order)
            new_result = self.get_drop_path_with_order(new_order)
            new_cost = cost_of_solution(self.graph, new_result[0], new_result[1], self.distanceLengthMemo)

            if new_cost < curr_min or random.random() < e_prob:
                curr_order = new_order
                curr_result = new_result
                curr_min = new_cost

        return curr_result


