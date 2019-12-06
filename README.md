# DTH - Drive the TAs Home

Drive the TAs Home is a NP-Hard problem that can be solved by using the following approximations in polynomial time.

  - Nearest Neighbour
  - Stiener Tree Approximation
  - Simulated Annealing


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
