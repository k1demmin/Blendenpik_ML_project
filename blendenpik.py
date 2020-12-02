#******************************************************************************
# CSCI 6961 Final Project
#   Author: Aaron Micah Green
#   Date: December 2, 2020
#******************************************************************************
import math
import numpy as np
import random as rand
import scipy.sparse.linalg as splalg
from sympy import fwht
from scipy.fftpack import dct

#******************************************************************************
# Blendenpik:
#   A linear system solver implementing the Blendenpik algorithm from the paper 
#   of the same title by Haim Avron, Petar Mayamounkov, and Sivan Toledo.
#******************************************************************************
class Blendenpik(object):
    
    #**************************************************************************
    # __init__:
    #   The constructor for Blendenpik.
    #
    #   Inputs:
    #       A - the matrix in the system Ax = b
    #       b - the vector in the system Ax = b
    #       transform (DEFAULT = 'DHT'): The method used for seed unitary transform
    #           in this instance of Blendenpik. There are three options:
    #           'DHT' - Discrete Harley Transform
    #           'DCT' - Discrete Cosine Transform
    #           'WHT' - Walsh-Hadamard Transform
    #       solver (DEFAULT = 'LSQR'): The method used for solving the system once
    #           preconditioning has been completed. There are three options:
    #           'LSQR' - 
    #           'LSMR' - 
    #       tolerance (DEFAULT = 1e-14) - how accurate the solution should be 
    #           after solving the system
    #       gamma (DEFAULT = 4) - a parameter controlling the probability that 
    #           any one row is sampled for the preconditioner
    #       maxIterations (DEFAULT = 3) - a parameter controlling how many attempts
    #           the algorithm will make to precondition the matrix to less than 
    #           a threshold for its condition number
    #************************************************************************** 
    def __init__(self, A, b, transform = 'DCT', solver = 'LSQR', tolerance = 1e-7, gamma = 10, maxIterations = 3):
        self.A = A
        self.b = b
        self.transform = transform
        self.solver = solver
        self.tolerance = tolerance
        self.gamma = gamma
        self.machineEps = np.finfo(float).eps
        self.maxIterations = maxIterations
        
        # Compute the number of rows to pad depending on the preconditioning alg:
        if transform == 'WHT':
            self.padding = int(math.pow(2, math.ceil(math.log2(A.shape[0]))))
        else:
            self.padding = math.ceil(A.shape[0] / 1000) * 1000
            
        # Create the padded matrix:
        self.M = np.zeros((self.padding, A.shape[1]))
        self.M[:A.shape[0], :A.shape[1]] = A
        
    #**************************************************************************
    # solve:
    #   Solve the system Ax = b using the Blendenpik algorithm. This involves
    #   running the selected preconditioning method, applying the preconditioner,
    #   and then solving with an iterative solver if the condition number post-
    #   preconditioning is below the desired threshold. Otherwise, use LAPACK.
    #
    #   Outputs:
    #       x - the solution to Ax = b
    #**************************************************************************
    def solve(self): 
        for iterations in range(self.maxIterations):
            D = createDiagonalMatrix(self.padding)
            self.M = D @ self.M;
            self.M = self.precondition()
            row_index = createRowSelector(self.padding, self.M.shape[1], self.gamma)
            sampledM = self.M[row_index,:]
            Q, R = np.linalg.qr(sampledM)
            conditionNumber = np.linalg.cond(R)
            if 1/conditionNumber > 5*self.machineEps:
                if self.solver == 'LSQR':
                    print("solving LSQR")
                    z = splalg.lsqr(self.A @ np.linalg.inv(R), self.b, atol = self.tolerance, btol = self.tolerance)[0]
                    x = splalg.lsqr(R, z)[0]
                elif self.solver == 'LSMR':
                    print("solving LSMR")
                    z = splalg.lsmr(self.A @ np.linalg.inv(R), self.b, atol = self.tolerance, btol = self.tolerance)[0]
                    x = splalg.lsmr(R, z)[0]
                return x
        print("Using Lapack")
        return np.linalg.lstsq(self.A, self.b)
    #**************************************************************************
    # precondition:
    #   Compute a preconditioner matrix for this instance's A-matrix based on 
    #   the selected preconditioner.
    #
    #   Outputs:
    #       C - the preconditioning matrix.
    #**************************************************************************
    def precondition(self):
        if self.transform == 'DHT':
            C = updated_DHT(self.M)
        elif self.transform == 'DCT':
            C = dct(self.M)
        elif self.transform == 'WHT':
            C = np.array(fwht(self.M))
        return C
    
#******************************************************************************
# createDiagonalMatrix:
#   Create a diagonal matrix of the specified dimension with plus/minus 1 on 
#   the diagonal, where each entry is equally likely to be positive or negative.
#   
#   Inputs:
#       dimension - the dimension of the matrix (it will be square)
#
#   Outputs:
#       D - a matrix meeting the above criteria
#******************************************************************************
def createDiagonalMatrix(dimension):
    D = np.eye(dimension)
    for i in range(dimension):
        if(rand.random() > 0.5): D[i,i] = -1
    return D

#******************************************************************************
# createRowSelector:
#   Create a vector to select which rows will be sampled for the preconditioner
#
#   Inputs:
#       m - the length of the vector
#       n - another dimension, whichexit should be less than m
#       gamma - a multiplier that can be used to scale the probability of a 1
#
#   Outputs:
#       S - the row selector indexing array
#******************************************************************************
def createRowSelector(m, n, gamma):
    S = np.zeros(m,dtype=np.int32)
    ratio = gamma * n / m
    for i in range(m):
        if(rand.random() < ratio): S[i] = i
    count = np.count_nonzero(S)
    while count < n:
        # if number of rows selected less than m
        first_zero_index = np.where(S == 0)[0][0]
        S[first_zero_index] = first_zero_index
        count += 1
    return S[S != 0]

def updated_DCT(A):
    N = A.shape[0]
    M = A.shape[1]
    print(N,M)
    C_DCT = np.zeros((N, M))
    for u in range(N):
        for v in range(M):
            C_DCT[u, v] = np.sqrt(2/N)*np.sqrt(2/M)* \
            np.sum(np.cos(np.pi*u*(1 + 2*np.arange(N-1))/2/N))* \
            np.sum(np.cos(np.pi*v*(1 + 2*np.arange(M-1))/2/M))*A[u,v]

    return C_DCT

def cas(angle):
    return np.cos(angle) + np.sin(angle)

def updated_DHT(A):
    N = A.shape[0]
    M = A.shape[1]
    C_DHT = np.zeros((N, M))
    for u in range(N):
        for v in range(M):
            sum = 0
            for n in range(N):
                for m in range(M):
                    angle = 2 * np.pi * (n + 1) * (u + 1) / N + 2 * np.pi * (m + 1) * (v + 1) / M
                    sum += A[n, m] * cas(angle)
            C_DHT[u, v] = np.round(sum, decimals=5)
    return C_DHT