from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit_algorithms import QAOA,NumPyMinimumEigensolver
import random


#Based off of https://qiskit-community.github.io/qiskit-optimization/tutorials/06_examples_max_cut_and_tsp.html
#Nice and easy to confirm solutions, but most of the math is abstracted away.

#insert graph here, for example I am using a complete graph with arbitrary weights
n=4
graph= nx.Graph()
graph.add_nodes_from(np.arange(0,n,1))
edgeList = [ (0,1,1.0),(0,2,3.0),(0,3,5.0),(1,2,4.0),(1,3,0.2),(2,3,0.1) ]
edgeDictionary={(0,1):1.0,(0,2):3.0,(0,3):5.0,(1,2):4.0,(1,3):0.2,(2,3):0.1,(1,0):1.0,(2,0):3.0,(3,0):5.0,(2,1):4.0,(3,1):0.2,(3,2):0.1}
#TODO: Check if minimum outdegree of all nodes is 2, then there is guaranteed to be a solution, otherwise there is no solution
#TODO: Automatically add edges to edgeDictionary so that you should only need to have nodes from u->v  and not v->u
graph.add_weighted_edges_from(edgeList)

#draw original graph
pos= nx.spring_layout(graph)
nx.draw_networkx(graph,node_size=300,pos=pos)
edge_labels = nx.get_edge_attributes(graph,"weight")
nx.draw_networkx_edge_labels(graph,edge_labels=edge_labels,pos=pos)
plt.show()

def connected(i, j):
    return (i,j) in edgeDictionary
#create a model
model = Model()
binary_var=model.binary_var_matrix(range(n),range(n))

#Minimize the sum of the distance from each city from timestep k to k+1
model.minimize(model.sum(edgeDictionary[(i, j)] * binary_var[(i, k)] * binary_var[(j, (k + 1) % n)] for i in range(n) for j in range(n) for k in range(n) if i!=j))

for i in range(n):
    model.add_constraint(model.sum(binary_var[(i, k)] for k in range(n)) == 1)
for k in range(n):
    model.add_constraint(model.sum(binary_var[(i, k)] for i in range(n)) == 1)
    
problem = from_docplex_mp(model)

algorithm_globals.random_seed = random.randint(0,100000000)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(problem)
#not sure why objective function value is incorrect but variable values show the correct matrix
print(result.prettyprint())