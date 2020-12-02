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
# Experiment for testing Blendenpik's performance relative to the aspect
# ratio of the system:
#**************************************************************************

# 'm' will represent the number of rows in the system, and will remain a
# constant for this experiment:
m = 30000;

# dataPoints will control how many ratios we test in this experiment:
dataPoints = 25;
ratios = [None] * dataPoints;
timesDHT = [None] * dataPoints;
timesDCT = [None] * dataPoints;
timesWHT = [None] * dataPoints;

# set bounds for the smallest and largest ratios to test:
smallestRatio = 100;
largestRatio = 20;

# Loop over the ratios we want to test:
for iteration in range(dataPoints):
    
    # Split the ratio range evenly:
    ratio = math.ceil(smallestRatio - (iteration) * (smallestRatio - largestRatio)/(dataPoints - 1));
    ratios[iteration] = ratio;
    n = int(m/ratio);
    
    # Generate a random matrix to test Blendenpik with:
    A = matrix_generate_coherent(rank=n, m=m, n=n, mu=0, sigma2=1, coherent=True, mu_li=0, num_li=0);
    
    # Set up the system for testing:
    x_true = np.random.rand(n);
    b = A @ x_true;
    blendenpikDHT = Blendenpik(A, b, 'DHT');
    blendenpikDCT = Blendenpik(A, b, 'DCT');
    blendenpikWHT = Blendenpik(A, b, 'WHT');
    
    # Test DHT:
    # We want to record how long the algorithm takes, so record the starting clock time:
    startTime = time.time();
    # Run the algorithm to solve the system:
    xDHT = blendenpikDHT.solve();
    # Compute the elapsed time:
    timesDHT[iteration] = time.time() - startTime;
    
    # Test DCT:
    # We want to record how long the algorithm takes, so record the starting clock time:
    startTime = time.time();
    # Run the algorithm to solve the system:
    xDCT = blendenpikDCT.solve();
    # Compute the elapsed time:
    timesDCT[iteration] = time.time() - startTime;
    
    # Test WHT:
    # We want to record how long the algorithm takes, so record the starting clock time:
    startTime = time.time();
    # Run the algorithm to solve the system:
    # xWHT = blendenpikWHT.solve();
    # Compute the elapsed time:
    timesWHT[iteration] = time.time() - startTime;
    
# Plot the results:
plt.plot(ratios, timesDHT, 'b-', ratios, timesDCT, 'g--')# , timesWHT);
plt.title("Speed Vs. Aspect Ratio")
plt.ylabel("Time (seconds)");
plt.xlabel("Ratio (m/x)")
plt.show();
