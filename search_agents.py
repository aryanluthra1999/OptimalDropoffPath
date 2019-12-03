import numpy as np
import utils
import networkx as netx
import student_utils as util170
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from pprint import pprint

def get_neighbours_and_weights(graph, location):
    ## return in the form of
    #################### FILL IN HERE ################
    #print(list(graph.nodes))
    nearby_neighbors=graph.neighbors(str(location))
    neighbours=[]
    weights = []
    for n in nearby_neighbors:
        weights+=[graph.edges[(str(location),str(n))]['weight']]
        neighbours+=[n]
    return neighbours, weights

class GameState():
    def __init__(self, homes_locations, location):
        self.location = location
        self.TA_left = len(homes_locations)
        self.homes_locations = homes_locations
        self.homes_reached = [False for _ in homes_locations]
        self.start = location
        self.path = []
        self.cost_so_far = 0

    def __eq__(self, o: object) -> bool:
        if type(o) == type(self):
            return self.__hash__() == o.__hash__()

        return super().__eq__(o)

    def __hash__(self) -> int:
        data = (tuple(self.location), self.TA_left, tuple(self.homes_reached))
        return hash(data)


    def copy(self):
        result = GameState(self.homes_locations, self.location)
        result.location = self.location
        result.TA_left = self.TA_left
        result.homes_locations = self.homes_locations.copy()
        result.homes_reached = self.homes_reached.copy()
        result.start=self.start
        result.path = self.path.copy()
        result.cost_so_far = self.cost_so_far
        return result

    def go_home_cost(self, graph):
        ### run dijkstras here to get home cost
        return 2/3 * netx.dijkstra_path_length(graph, self.location, self.start, weight='weight')

    def get_legal_actions(self, graph):

        if self.TA_left == 0:
            cost = self.go_home_cost(graph)
            result = [("go_home", cost)]

            # print(self.path, self.cost_so_far, cost)
            #
            # print(self.TA_left)
            # print(self.homes_reached)
            # print(self.location, "\n")


            return result


        if self.location in self.homes_locations and self.homes_reached[self.homes_locations.index(self.location)] == False:
            return [("drop", 0)]

        dropoff_cost = self.get_dropoff_cost_and_loc(graph)[0]
        result = [("drop", dropoff_cost)]

        neighbours, weights = get_neighbours_and_weights(graph, self.location)

        for i in range(len(neighbours)):
            r = (neighbours[i], 2/3*weights[i])
            result.append(r)

        return result

    def get_dropoff_cost_and_loc(self,G):
        ## returned tuple is of the form:
        ## (cost to nearest home, index of nearest home in self.homes_locations)
        ################ FILL IN HERE #################
        costs=[]
        for i in range(len(self.homes_locations)):
            home = self.homes_locations[i]
            if self.homes_reached[i] == True:
                costs += [float("inf")]
            else:
                costs +=[netx.dijkstra_path_length(G, str(self.location), str(home), weight='weight')]

        return (min(costs),np.argmin(np.array(costs)))

    def successor(self, action, cost, G):
        new_gs = self.copy()

        if action == "drop":
            new_gs.TA_left = new_gs.TA_left - 1
            new_gs.homes_reached[int(self.get_dropoff_cost_and_loc(G)[1])] = True
        elif action == 'go_home':
            new_gs.location = self.start
        else:
            ############# FILL IN HERE ############
            new_gs.location = action

        new_gs.cost_so_far += cost
        new_gs.path.append(action)

        return new_gs, action, cost

    def getSuccessors(self, graph):
        actions = self.get_legal_actions(graph)
        result = []

        for a in actions:
            r = self.successor(a[0], a[1],graph)
            result.append(r)

        return result

    def isGoalState(self):
        #result = self.TA_left == 0
        result = sum(np.array(self.homes_reached) == False) == 0
        result = result and self.location == str(self.start)  ### FIX IF NESSECARY

        print(self.TA_left)
        print(self.homes_reached)
        print(self.location, "\n")

        return result


class SearchAgent():

    def __init__(self, adj_matrix, homes_arr, soda_loc,locs):
        self.start_state = GameState(homes_arr, soda_loc)
        self.graph = util170.adjacency_matrix_to_graph(adj_matrix)[0]  ## maybe we want a graph object from network x instead
        mapping = dict(zip(self.graph,locs))
        self.graph = netx.relabel_nodes(self.graph, mapping)



    def uniformCostSearch(self):
        """Search the node of least total cost first."""
        # path, weights = {}, {}
        closed = set()
        fringe = utils.PriorityQueue()
        start = self.start_state
        fringe.push((start, None, 0), 0)
        goal = None

        # path[(start, None, 0)] = None
        # weights[(start, None, 0)] = 0

        while not fringe.isEmpty():
            curr_state = fringe.pop()
            state = curr_state[0]

            if state.isGoalState():
                goal = curr_state
                break

            if state not in closed:
                closed.add(state)
                successors = state.getSuccessors(self.graph)
                for next_state in successors:
                    if next_state[0] not in closed:
                        fringe.push(next_state, next_state[0].cost_so_far)

        # result = []
        # while goal[1] != None:
        #     result.append(goal)
        #     goal = path[goal]
        # result = result[::-1]
        # #print(result)
        #
        #
        # print("\n SUM: ", sum([r[2] for r in result]))

        print("Nodes expanded: ", len(closed))
        print("Path: ", goal[0].path)
        print("cost: ", goal[0].cost_so_far)

        return goal

    def nullHeuristic(self, state):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def naiveHeuristic(self, state):
        return max(0, state.TA_left - 1)

    def mstHeuristic(self, state):
        locations = []

        for i in range(len(state.homes_reached)):
            if state.homes_reached[i]:
                continue
            else:
                locations.append(state.homes_locations)


        homes_matrix = np.array([[0 for _ in locations] for _ in locations])
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                homes_matrix[i, j] = self.distance(locations[i], locations[j])
        homes_matrix = csr_matrix(homes_matrix)
        Tcsr = minimum_spanning_tree(homes_matrix).toarray().astype(float)

        result = sum(Tcsr)
        result += self.distance(state.location, self.start_state) + state.get_dropoff_cost_and_loc(self.graph)

        return result


    def distance(self, loc1, loc2):
        ## new dijkstras method but with memoized distances
        ## will reduce heuristic time immensely
        ## copy this method into GameState as well and use it for our method: GameState.get_dropoff_cost_and_loc
        ########################### Fill in here ##############
        return 1

    def astar(self, heuristic=mstHeuristic):
        """Search the node of least total cost first."""
        # path, weights = {}, {}
        closed = set()
        fringe = utils.PriorityQueue()
        start = self.start_state
        fringe.push((start, None, 0), heuristic(self, start))
        goal = None

        while not fringe.isEmpty():
            curr_state = fringe.pop()
            state = curr_state[0]

            if state.isGoalState():
                goal = curr_state
                break

            if state not in closed:
                closed.add(state)
                successors = state.getSuccessors(self.graph)
                for next_state in successors:
                    if next_state[0] not in closed:
                        fringe.push(next_state, next_state[0].cost_so_far + heuristic(self, next_state[0]))

        print("Nodes expanded: ", len(closed))
        print("Path: ", goal[0].path)
        print("cost: ", goal[0].cost_so_far)

        return goal[0].path
