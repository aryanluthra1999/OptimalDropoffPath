# DTH - Drive the TAs Home

Drive the TAs Home is a NP-Hard problem that can be solved by using the following approximations in polynomial time.

  - Nearest Neighbour
  - Stiener Tree Approximation
  - Simulated Annealing


### Steps Involved
There are several approximations used in order to make the solution better over time:

0) We started off with the exact Gamestate solution and ran it for 2 minutes and switch to Nearest Neighbour approximation if we were not able to find the exact solution.
1) Use Nearest Neighbour to find a determinictic solution for all graphs.
2) Improve upon it by running Steiner Approximation and update the solution if it is better.
3) Run Simulated Annealing for 270k interations to find a local minimum for each input.

### Note:
The outputs are not always reproduced exactly because Steiner Approximation is a random.
In the solve method in solver.py the different approximation are present, please uncoment the approximation that you want to use.

### Installation

DTH requires [Nteworkx](https://networkx.github.io/documentation/stable/install.html) to run.

Install the dependencies and devDependencies and start the program.

```sh
$ pip install networkx
$ pip install numpy
$ pip install tqdm
```
For multi-threading we use [multiprocessing](https://pypi.org/project/multiprocess/):

```
$ pip install multiprocess
```

#### Building for program:
To start solving the inputs:
```sh
$ python solver.py --all inputs outputs
```
Generate output.json
```sh
$ python compress_output.py outputs/
```
