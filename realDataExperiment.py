#******************************************************************************
# CSCI 6961 Final Project
#   Author: Aaron Micah Green
#   Date: December 2, 2020
#******************************************************************************

import numpy as np;
import time;
import math;
import matplotlib.pyplot as plt;
from blendenpik import Blendenpik;
import scipy.sparse.linalg as splalg

from keras.datasets import cifar100, cifar10

#**************************************************************************
# Helper function for plotting from https://heartbeat.fritz.ai/using-least-squares-based-classification-to-detect-digits-c5146c807b2c
#**************************************************************************

def plot_cm(y_true, y_pred, classes, cm, title=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#**************************************************************************
# Experiment for testing Blendenpik's performance versus Lapack on a real data set.
#**************************************************************************
print(" -- experiment on CIFAR-10 -- ")
# Load CIFAR 100 from Keras API, there are 100 target classes for the classification task
(trainX, trainY), (testX, testY) = cifar10.load_data()
n_train = trainX.shape[0]
n_test = testX.shape[0]

n_class = 10

'''
#**************************************************************************
# we may merge train and test together for a larger system which will have 60000 rows in total
#**************************************************************************
data = np.concatenate((trainX, testX), axis = 0)
target = np.concatenate((trainY, testY), axis = 0)
n_data = data.shape[0]
print('data=%s, target=%s' % ( data.shape, target.shape))
'''

# flatten the 3 channel RGB image to be 1-D array
flat_trainX = trainX.flatten().reshape(n_train, 32*32*3)
flat_testX = testX.flatten().reshape(n_test, 32*32*3)

#print('Train: X=%s, y=%s' % ( trainX.shape, trainY.shape))
#print('Test: X=%s, y=%s' % ( testX.shape, testY.shape))
print('flat_Train: X=%s, y=%s' % ( flat_trainX.shape, trainY.shape))
print('flat_Test: X=%s, y=%s' % ( flat_testX.shape, testY.shape))


# Sample image output
'''
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i])
plt.show()
'''

#**************************************************************************
# Perform pure LSQR, Blendenpik w/ LSQR and BLendenpik w/ LSMR on the system
#**************************************************************************

A = flat_trainX
b = trainY
blendenpikLSQR = Blendenpik(A, b, solver = 'LSQR');
blendenpikLSMR = Blendenpik(A, b, solver = 'LSMR');

#**************************************************************************
# Test pure LSQR
#**************************************************************************
print("-- starting testing with pure LSQR -- ")
startTime = time.time();
Q, R = np.linalg.qr(A)
conditionNumber = np.linalg.cond(R)
if 1/conditionNumber > 5*np.finfo(float).eps:
    z = splalg.lsqr(A @ np.linalg.inv(R), b, atol = 1e-7, btol = 1e-7)[0]
    xPureLSQR = splalg.lsqr(R, z)[0]
timesPureLSQR = time.time() - startTime;

# perform prediction
resultLabels = flat_testX @ xPureLSQR
resultLabels[resultLabels < 0] = 0
resultLabels[resultLabels > (n_class-1)] = (n_class - 1)
resultLabels = np.floor(resultLabels).astype(int)

# cm is the confusion matrix of the prediction
cm = np.zeros([n_class, n_class])
for i in range(len(testY)):
    cm[int(testY[i])][int(resultLabels[i])] += 1
cm = cm.astype(int)

# Plot the confusion matrix
np.set_printoptions(precision=2)
class_names = []
for i in range(n_class):
    class_names.append(str(i))
plot_cm(testY, resultLabels, classes=class_names, cm = cm,
                      title='Normalized confusion matrix for pure LSQR on CIFAR-10')
plt.show()



#**************************************************************************
# Test Blendenpik w/ LSQR:
#**************************************************************************
print("-- starting testing with BLendenpik w/ LSQR -- ")
startTime = time.time();
# Run the algorithm to solve the system:
xLSQR = blendenpikLSQR.solve();
# Compute the elapsed time:
timesLSQR = time.time() - startTime;

# perform prediction
resultLabels = flat_testX @ xLSQR
resultLabels[resultLabels < 0] = 0
resultLabels[resultLabels > (n_class-1)] = (n_class-1)
resultLabels = np.floor(resultLabels).astype(int)

# cm is the confusion matrix of the prediction
cm = np.zeros([n_class, n_class])
for i in range(len(testY)):
    cm[int(testY[i])][int(resultLabels[i])] += 1
cm = cm.astype(int)

# Plot the confusion matrix
np.set_printoptions(precision=2)
class_names = []
for i in range(n_class):
    class_names.append(str(i))

plot_cm(testY, resultLabels, classes=class_names, cm = cm,
                      title='Normalized confusion matrix For Blendenpik w/ LSQR on CIFAR-10')
plt.show()


#**************************************************************************
# Test Blendenpik w/ LSMR:
#**************************************************************************
print("-- starting testing with BLendenpik w/ LSMR -- ")
startTime = time.time();
# Run the algorithm to solve the system:
xLSMR = blendenpikLSMR.solve();
# Compute the elapsed time:
timesLSMR = time.time() - startTime;

# perform prediction
resultLabels = flat_testX @ xLSMR
resultLabels[resultLabels < 0] = 0
resultLabels[resultLabels > (n_class-1)] = (n_class-1)
resultLabels = np.floor(resultLabels).astype(int)

# cm is the confusion matrix of the prediction
cm = np.zeros([n_class, n_class])
for i in range(len(testY)):
    cm[int(testY[i])][int(resultLabels[i])] += 1
cm = cm.astype(int)

# Plot the confusion matrix
np.set_printoptions(precision=2)
class_names = []
for i in range(n_class):
    class_names.append(str(i))

plot_cm(testY, resultLabels, classes=class_names, cm = cm,
                      title='Normalized confusion matrix for Blendenpik w/ LSMR on CIFAR-10')
plt.show()

#**************************************************************************
# Time performance with different solvers 
#**************************************************************************
print(" -- time performance -- ")
print('Time for solving with PureLSQR: %s' % (timesPureLSQR))
print('Time for solving with LSQR: %s' % (timesLSQR))
print('Time for solving with LSMR: %s' % (timesLSMR))


'''
#**************************************************************************
# Perform the same experiment on CIFAR-100
#**************************************************************************
# just change the loaded dataset to be CIFAR-100, and keep track of the dimensions 
'''