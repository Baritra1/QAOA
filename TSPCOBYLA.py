from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import DiagonalGate
from qiskit import transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import math

import random


def connected(i, j):
    return (i,j) in edgeDictionary
#Create quantum circuit for QAOA from edges and parmeters
def QAOA(n,edges,p,betas,gammas):  
    #Define quantum and classical registers
    nQubits=n**2
    qr = QuantumRegister(nQubits)
    cr = ClassicalRegister(nQubits)
    circuit = QuantumCircuit(qr,cr)
    
    #Initial Hadamards
    for q in range(nQubits):
        circuit.h(q)        
    #For the number of specified iterations
    for P in range(p):

        #cost hamiltonian
        for i in range(n):
            for j in range(n):
                for k in range(n-1):
                    if connected(i,j):
                        # print(k*n+1)
                        # print(k*(n+1)+j)
                        qubitOne=k*n+i
                        qubitTwo=(k+1)*n+j
                        circuit.cx(qubitOne,qubitTwo)
                        circuit.rz(phi=gammas[P]*edgeDictionary[(i,j)],qubit=qubitTwo)
                        circuit.cx(qubitOne,qubitTwo)
        maxEdgeWeight=0
        for j in edges:
            maxEdgeWeight=max(maxEdgeWeight,j[2])
        penalty=20*maxEdgeWeight
        #generate penalty unitary matrices 
        entries= [1]*(2**n)
        cur=1
        while cur<2**n:
            entries[cur]=0
            cur*=2
        # print(entries)
        for i in range(len(entries)):
            entries[i]=math.e**(entries[i]*gammas[P]*penalty*1j)

        #being in 2 cities at the same time
        for t in range(n):
            excessCity = list(range(t*n,(t+1)*n))
            circuit.append(DiagonalGate(entries),excessCity)

        #visiting the same city multiple times
        for i in range(n):
            excessTime=[]
            cur=i
            for j in range(n):
                excessTime.append(cur)
                cur+=n
            circuit.append(DiagonalGate(entries),excessTime)

        #mixer hamiltonian
        for q in range(nQubits):
            circuit.rx(theta=2*betas[P],qubit=q)   
    circuit.measure(qr,cr)
    return circuit

#Compute all scores for a set of edges
def computeExpectationValue(counts,edges,n):
    totalScore = 0
    totalSamples = 0
    #For each bitstring measured (keys in counts dictionary)
    for bitstring in counts.keys():
        score = 0  #Score for this bitstring
        substrings=[bitstring[i:i + n] for i in range(0, len(bitstring), n)]
        # print(substrings)
        maxEdgeWeight=0
        for j in edges:
            maxEdgeWeight=max(maxEdgeWeight,j[2])
        #penalty for violating constraints
        penalty=20*maxEdgeWeight

        #apply penalty for visiting same city more than once
        for j in range(n):
            cur=j
            count=0
            for k in range(n):
                if( bitstring[cur]=='1'):
                    count+=1
                cur+=n
            if count!=1:
                score+=penalty
            
        #apply penalty for visiting multiple cities at same time
        for j in substrings:
            count=0
            for k in j:
                if(k=='1'):
                    count+=1
            if count!=1:
                score+=penalty
    
        #add weight of edge
        for j in range(1,len(substrings)):
            u=0
            v=0
            for k in substrings[j-1]:
                if k!='1':
                    u+=1
                else:
                    break
            for k in substrings[j]:
                if k!='1':
                    v+=1
                else:
                    break
            if connected(u,v):
                score+=edgeDictionary[(u,v)]
            else:
                score+=penalty

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
    score = computeExpectationValue(counts,edges,n)
    return(score)

def optimize(n,edges,p,nIterations,nSamples,params):
    out = minimize(ExpectationValue,x0=params,method="COBYLA",options={'maxiter':nIterations})
    print(f'Out: {out}')
    optimal_params = out['x'] 
    counts=runCKT(optimal_params)
    # plt.bar(list(counts.keys()),list(counts.values()),width=0.9)
    # plt.xlabel("bitstrings")
    # plt.ylabel("counts")
    # plt.title("Results")
    # plt.show()
    return max(zip(counts.values(), counts.keys()))[1]


#4 qubits
n = 4
#Edges of the maxcut problem
#make sure to change graph later
edges = [ (0,1,1.0),(0,2,3.0),(0,3,5.0),(1,2,4.0),(1,3,0.2),(2,3,0.1) ]
edgeDictionary={(0,1):1.0,(0,2):3.0,(0,3):5.0,(1,2):4.0,(1,3):0.2,(2,3):0.1,(1,0):1.0,(2,0):3.0,(3,0):5.0,(2,1):4.0,(3,1):0.2,(3,2):0.1}
#TODO: Automatic addition of second half of edges
#p=2 is sufficient for this problem
p = 4
#A sufficient number of optimization iterations to solve problem
nIterations = 200
#Typically need quite a few samples (measurements of quantum circuit) per iteration to 
nSamples = 10000
#Heuristically chosen a and c
params=[]
for i in range(p*2):
    params.append(0.01*np.random.rand())
print("Best bitstring:",optimize(n=n,edges=edges,p=p,nIterations=nIterations,nSamples=nSamples,params=params))