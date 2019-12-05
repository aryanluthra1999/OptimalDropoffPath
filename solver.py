import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils
import search_agents as search
import orderApproximators
import SteinerApproxSolver
import networkx as netx
from output_validator.py import tests
import time
from student_utils import cost_of_solution
from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    # list_of_locations=[str(i) for i in list_of_locations]
    #
    # searchAgent = search.SearchAgent(adjacency_matrix,list_of_homes,starting_car_location,list_of_locations)
    #
    # result = searchAgent.astar()

    #order_approx_agent = orderApproximators.OrderApproximator(adjacency_matrix, list_of_homes, starting_car_location, list_of_locations)

    graph = adjacency_matrix_to_graph(adjacency_matrix)[0]  ## maybe we want a graph object from network x instead
    mapping = dict(zip(graph, list_of_locations))
    graph = netx.relabel_nodes(graph, mapping)
    #result = order_approx_agent.bootstrap_approx()
    from student_utils import cost_of_solution
    #print(result)

    steiner_approx_solver = SteinerApproxSolver.SteinerApproxSolver(adjacency_matrix, list_of_homes, starting_car_location, list_of_locations)
    brr=steiner_approx_solver.solveSteinerTreeDTH()
    steiner_approx_solver_order = cost_of_solution(graph,brr[0],brr[1])
    print(steiner_approx_solver_order)

def runSolver(inputFile):
    input_data = utils.read_file(inputFile)
    params=[]
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)

def compareSolution(fileName,sol):
    better=False
    input_file =  utils.read_file(fileName)
    if path.exists(fileName[0:fileName.index('.')+1]+'out'):
        return better
    else:
        output_file = utils.read_file(fileName[0:filename.index('.')+1]+'out')
        if costs(input_file,output_file,[])<sol):
            better=True
        return better


"""if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)"""


if __name__ == "__main__":
    fname = "inputs/7_50.in"
    runSolver(fname)
