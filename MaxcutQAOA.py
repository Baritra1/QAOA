from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile
from qiskit_aer import Aer
import matplotlib.pyplot as plt

import random

# Inspired by https://arxiv.org/pdf/2106.01578, but generalized
# to work on a weighted graph, and this method is also used to solve TSP 
# in TravelingSalesmanQAOA.py.

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
                score = score + edges[j][2]
        totalScore += score * counts[bitstring]  #Multiply score times the # of times it was observed
        totalSamples += counts[bitstring]        #Keep track of the number of measurements (samples)
    return(totalScore/totalSamples)

#Run the circuit and return counts
def runCKT(circuit,shots,noise=False):
    #Had to update this code
    #TODO: Add noise simulation
    simulator = Aer.get_backend('qasm_simulator')
    transpiled = transpile(circuit,backend=simulator)
    counts=simulator.run(transpiled,shots=shots).result().get_counts(circuit)  
    return counts

#Run a circuit and get expectation value
def ExpectationValue(circuit,edges,nSamples):
    #Run circuit and collect counts
    counts = runCKT(circuit=circuit,shots=nSamples)
    # print(counts)
    #Get the score of the counts
    score = computeExpectationValue(counts,edges)
    return(score)

def SPSAforQAOA(n,edges,p,nIterations,nSamples,a_start,c_start,decay):
    #Initiate
    a = []; c = [];
    for i in range(1,nIterations+1):
        a.append( a_start / (i ** decay) )
        c.append( c_start / (i ** decay) )
    for i in range(nIterations):
        if( c[i] < 0.01 ):
            c[i] = 0.01
            
    #Initiate gamma,beta
    gammas = []
    betas  = []
    for P in range(p):
        gammas.append( random.uniform(-.1,.1) )
        betas.append(  random.uniform(-.1,.1) )
    
    #Run iterations of SPSA
    for i in range(nIterations):
        #Randomly perturb gammas,betas by c[i]
        Delta_gammas = []; gammas_plus = []; gammas_minus = [];
        Delta_betas  = []; betas_plus  = []; betas_minus  = [];
        for P in range(p):
            #Generate perturbation vectors of bernoulli variables with magnitude c[i]
            Delta_gammas.append( random.randrange(2)*2 - 1 * c[i] )
            Delta_betas.append(  random.randrange(2)*2 - 1 * c[i] )
            #Create +/- versions of the parameters
            gammas_plus.append( gammas[P] + Delta_gammas[P])
            gammas_minus.append(gammas[P] - Delta_gammas[P])
            betas_plus.append( betas[P] + Delta_betas[P])
            betas_minus.append(betas[P] - Delta_betas[P])                
         
        #Get the circuits for the +/- versions
        pCircuit = QAOA(nQubits=n,edges=edges,p=p,betas=betas_plus,gammas=gammas_plus)
        mCircuit = QAOA(nQubits=n,edges=edges,p=p,betas=betas_minus,gammas=gammas_minus)
        #Run the +/- versions
        Fplus =  ExpectationValue(circuit=pCircuit,edges=edges,nSamples=nSamples)  
        Fminus = ExpectationValue(circuit=mCircuit,edges=edges,nSamples=nSamples)  
        
        #Compute estimated gradients 
        g_gammas = []; g_betas = [];
        for P in range(p):
            g_gammas.append( (Fplus - Fminus) / (2*Delta_gammas[P]) )
            g_betas.append(  (Fplus - Fminus) / (2*Delta_betas[P])  )
            
        #Update the parameters
        for P in range(p):
            gammas[P] = gammas[P] + a[i]*g_gammas[P]
            betas[P]  = betas[P]  + a[i]*g_betas[P]
        
        #Report progress
        print('Iteration:',i,'Exp(+):',Fplus,'Exp(-):',Fminus)
    counts=runCKT(circuit=QAOA(nQubits=n,edges=edges,p=p,betas=betas,gammas=gammas),shots=nSamples)
    plt.bar(list(counts.keys()),list(counts.values()),width=0.9)
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.title("Results")
    plt.show()
    return max(zip(counts.values(), counts.keys()))[1]


#4 qubits
n = 4
#Edges of the maxcut problem
edges = [ (0,1,1.0),(0,2,3.0),(0,3,1.0),(1,2,4.0),(1,3,2.0),(2,3,1.0) ]
#p=2 is sufficient for this problem
p = 2
#A sufficient number of optimization iterations to solve problem
nIterations = 200
#Typically need quite a few samples (measurements of quantum circuit) per iteration to 
nSamples = 10000
#Heuristically chosen a and c
a_start = 0.25
c_start = 0.25
decay = 0.5
print("Best bitstring:",SPSAforQAOA(n=n,edges=edges,p=p,nIterations=nIterations,nSamples=nSamples,a_start=a_start,c_start=c_start,decay=decay))