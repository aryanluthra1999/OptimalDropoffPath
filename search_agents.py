import numpy as np
import utils
import scipy as sp
import networkx

def get_neighbours_and_weights(graph, location):
    ## return in the form of
    #################### FILL IN HERE ################
    neighbours= []
    weights = []
    return neighbours, weights

class GameState():
    def __init__(self, homes_locations, location = "Soda"):
        self.location = location
        self.TA_left = len(homes_locations)
        self.homes_locations = homes_locations
        self.homes_reached = [False for _ in homes_locations]
        #self.path = []
        #self.cost_so_far = 0


    def copy(self):
        result = GameState(self.homes_locations, self.location):
        result.location = self.location
        result.TA_left = self.TA_left
        result.homes_locations = self.homes_locations
        result.homes_reached = self.homes_reached
        #result.path = self.path
        #result.cost_so_far = self.cost_so_far

        return result

    def go_home_cost(self):
        ### run dijkstras here to get home cost
        return 10

    def get_legal_actions(self, graph):

        if self.TA_left == 0:
            cost = self.go_home_cost()
            result = [("go_home", cost)]
            return result

        if self.location in self.homes_locations:
            return [("drop", 0)]

        dropoff_cost = self.get_dropoff_cost_and_loc()[0]
        result = [("drop", dropoff_cost)]

        neighbours, weights = get_neighbours_and_weights(graph, self.location)

        for i in range(len(neighbours)):
            r = (neighbours[i], 2/3*weights[i])
            result.append(r)

        return result

    def get_dropoff_cost_and_loc(self):
        ## returned tuple is of the form:
        ## (cost to nearest home, index of nearest home in self.homes_locations)
        ################ FILL IN HERE #################
        return 1, 0

    def successor(self, action, cost):
        new_gs = self.copy()

        if action == "drop":
            new_gs.TA_left -= 1
            new_gs.homes_reached[self.get_dropoff_cost_and_loc()[1]] = True
        else:
            ############# FILL IN HERE ############

        #new_gs.cost_so_far += cost
        #new_gs.path.append(action)
        return new_gs

    def getSucessors(self, graph):
        actions = self.get_legal_actions(graph)
        result = []

        for a in actions:
            r = self.successor(a[0], a[1]), a[1]
            result.append(r)

        return result

    def isGoalState(self):
        result = self.TA_left == 0
        result = result and sum(np.array(self.homes_reached) == False) == 0
        result = result and self.location == "Soda"  ### FIX IF NESSECARY

        return result


class SearchAgent():

    def __init__(self, adj_matrix, homes_arr, soda_loc):
        self.start_state = GameState(homes_arr, soda_loc)
        self.graph = adj_matrix  ## maybe we want a graph object from network x instead

    def uniformCostSearch(self, problem):
        """Search the node of least total cost first."""
        closed = set()
        weights = {}
        fringe = utils.PriorityQueue()
        start = self.start_state
        fringe.push((start, None, 0), 0)
        goal = None
        weights[(start, None, 0)] = 0

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
                        path[next_state] = curr_state
                        weights[next_state] = weights[curr_state] + next_state[2]
                        fringe.push(next_state, weights[next_state])

        #print(goal.path)
        return goal

    def nullHeuristic(state, problem=None):
        """
        A heuristic function estimates the cost from the current state to the nearest
        goal in the provided SearchProblem.  This heuristic is trivial.
        """
        return 0

    def aStarSearch(problem, heuristic=nullHeuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        path, weights = {}, {}
        closed = set()
        fringe = util.PriorityQueue()
        start = problem.getStartState()
        fringe.push((start, None, 0), heuristic(start, problem))
        goal = None
        path[(start, None, 0)] = None
        weights[(start, None, 0)] = 0
        while not fringe.isEmpty():
            curr_state = fringe.pop()
            state = curr_state[0]

            if problem.isGoalState(state):
                goal = curr_state
                break

            if state not in closed:
                closed.add(state)
                successors = problem.getSucessors(state)
                for next_state in successors:
                    if next_state[0] not in closed:
                        path[next_state] = curr_state
                        weights[next_state] = weights[curr_state] + next_state[2]
                        fringe.push(next_state, weights[next_state] + heuristic(next_state[0], problem))

        result = []
        while goal[1] != None:
            result.append(goal[1])
            goal = path[goal]
        result = result[::-1]
        print(result)
        return result



