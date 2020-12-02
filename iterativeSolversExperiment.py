#******************************************************************************
# CSCI 6961 Final Project
#   Author: Aaron Micah Green
#   Date: December 2, 2020
#******************************************************************************

import numpy as np;
from matrixgen import matrix_generate_coherent;
import time;
import math;
import matplotlib.pyplot as plt;
from blendenpik import Blendenpik;
    
#**************************************************************************
# Experiment for testing Blendenpik's performance with different solvers
#**************************************************************************
# dataPoints will control how many sizes of systems we test in this experiment:
dataPoints = 10;
sizes = [None] * dataPoints;
timesLSQR = [None] * dataPoints;
timesLSMR = [None] * dataPoints;

# set the bounds on the system size that we want to test:
smallestSystem = 10000;
largestSystem = 50000;

# Choose an aspect ratio for the systems we will be testing:
systemAspectRatio = 40;

# Loop over the sizes we want to test:
itrmax=dataPoints
for iteration in range(1,itrmax,1):
    print("dtpoints:",iteration)
    # Split the size range evenly:
    size = math.ceil(smallestSystem + iteration * (largestSystem - smallestSystem)/(dataPoints - 1));
    sizes[iteration] = size;
    n = int(size/systemAspectRatio);
    print('matrix size',size,n)
    # Generate a random matrix to test Blendenpik with:
    A = matrix_generate_coherent(rank=n, m=size, n=n, mu=0, sigma2=1, coherent=True, mu_li=0, num_li=0);
    print("matrix generated")
    # Set up the system for testing:
    x_true = np.random.rand(n);
    b = A @ x_true;
    blendenpikLSQR = Blendenpik(A, b, solver = 'LSQR');
    blendenpikLSMR = Blendenpik(A, b, solver = 'LSMR');
    # Test LSQR:
    # We want to record how long the algorithm takes, so record the starting clock time:
    startTime = time.time();
    # Run the algorithm to solve the system:
    xLSQR = blendenpikLSQR.solve();
    # Compute the elapsed time:
    timesLSQR[iteration] = time.time() - startTime;
    
    # Test LSMR:
    # We want to record how long the algorithm takes, so record the starting clock time:
    startTime = time.time();
    # Run the algorithm to solve the system:
    xLSMR = blendenpikLSMR.solve();
    # Compute the elapsed time:
    timesLSMR[iteration] = time.time() - startTime;
    
# Plot the results:
plt.plot(sizes[:dataPoints-1], timesLSQR[:dataPoints-1], 'b-', label = "LSQR");
plt.plot(sizes[:dataPoints-1], timesLSMR[:dataPoints-1], 'g--', label = "LSMR")
plt.title("Speed Vs. Size (by solver)")
plt.ylabel("Time (seconds)");
plt.xlabel("System Size")
plt.legend()
plt.show();
