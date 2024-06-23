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
edges=[(0,1,1.0),(0,2,3.0),(0,3,1.0),(1,2,4.0),(1,3,2.0),(2,3,1.0) ]
graph.add_weighted_edges_from(edges)

#draw original graph
pos= nx.spring_layout(graph)
nx.draw_networkx(graph,node_size=300,pos=pos)
edge_labels = nx.get_edge_attributes(graph,"weight")
nx.draw_networkx_edge_labels(graph,edge_labels=edge_labels,pos=pos)
plt.show()


#create a model
model = Model()
binary_var=model.binary_var_list(n)

#Maximize C(x)=Σ_{i,j} w_{ij} x_i (1-x_j)+Σ_i w_i x_i
model.maximize(model.sum(w * binary_var[i] * (1 - binary_var[j]) + w * (1 - binary_var[i]) * binary_var[j] for i, j, w in edges))

problem = from_docplex_mp(model)

algorithm_globals.random_seed = random.randint(0,100000000)
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
result = exact.solve(problem)
print(result.prettyprint())