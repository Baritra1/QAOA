from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import Aer
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

# Uses scipy to optimize rather than SPSA...this works so much more consistently
#than SPSA.

#Create quantum circuit for QAOA from edges and parmeters
def QAOA(nQubits,edges,p,betas,gammas):  
    #Define quantum and classical registers
    qr = QuantumRegister(nQubits)
    cr = ClassicalRegister(nQubits)
    circuit = QuantumCircuit(qr,cr)
    
    #Initial Hadamards
    for q in range(nQubits):
        circuit.h(q)        
    #For the number of specified iterations
    for P in range(p):
        for j in range(len(edges)):
            circuit.cx(edges[j][0],edges[j][1])
            #Caused a lot of annoyance, turns out the rotation angle needs to be weighted as well         
            circuit.rz(phi=gammas[P]*edges[j][2],qubit=edges[j][1])           
            circuit.cx(edges[j][0],edges[j][1])
        
        for q in range(nQubits):
            circuit.rx(theta=2*betas[P],qubit=q)   
    circuit.measure(qr,cr)
    return circuit

#Compute all scores for a set of edges
def computeExpectationValue(counts,edges):
    totalScore = 0
    totalSamples = 0
    #For each bitstring measured (keys in counts dictionary)
    for bitstring in counts.keys():
        score = 0  #Score for this bitstring
        for j in range(len(edges)):
            if( bitstring[edges[j][0]] != bitstring[edges[j][1]] ):
                score += edges[j][2]
        totalScore += score * counts[bitstring]  #Multiply score times the # of times it was observed
        totalSamples += counts[bitstring]        #Keep track of the number of measurements (samples)
    print("Cost: "+str(totalScore))
    return(totalScore/totalSamples)

#Run the circuit and return counts
def runCKT(params):
    #TODO: Add noise simulation
    simulator = Aer.get_backend('qasm_simulator')
    shots=nSamples
    betas=[]
    gammas=[]
    for i in range(len(params)):
        if(i<len(params)/2):
            betas.append(params[i])
        else:
            gammas.append(params[i])
    circuit=QAOA(n,edges,p,betas,gammas)
    transpiled = transpile(circuit,backend=simulator)
    counts=simulator.run(transpiled,shots=shots).result().get_counts(circuit)  
    return counts

#Run a circuit and get expectation value
def ExpectationValue(params):
    #Run circuit and collect counts
    counts = runCKT(params)
    # print(counts)
    #Get the score of the counts
    score = computeExpectationValue(counts,edges)
    return(-score)

def optimize(n,edges,p,nIterations,nSamples,params):
    out = minimize(ExpectationValue,x0=params,method="COBYLA",options={'maxiter':nIterations})
    print(f'Out: {out}')
    optimal_params = out['x'] 
    counts=runCKT(optimal_params)
    plt.bar(list(counts.keys()),list(counts.values()),width=0.9)
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.title("Results")
    plt.show()
    return max(zip(counts.values(), counts.keys()))[1]


#4 qubits
n = 4
#Edges of the maxcut problem
edges = [(0,1,1.0),(0,2,3.0),(0,3,1.0),(1,2,4.0),(1,3,2.0),(2,3,1.0) ]
#p=2 is sufficient for this problem
p = 4
#A sufficient number of optimization iterations to solve problem
nIterations = 500
#Typically need quite a few samples (measurements of quantum circuit) per iteration to 
nSamples = 10000
#Heuristically chosen a and c
params=[]
for i in range(p*2):
    params.append(0.01*np.random.rand())
print(params)
print("Best bitstring:",optimize(n=n,edges=edges,p=p,nIterations=nIterations,nSamples=nSamples,params=params))