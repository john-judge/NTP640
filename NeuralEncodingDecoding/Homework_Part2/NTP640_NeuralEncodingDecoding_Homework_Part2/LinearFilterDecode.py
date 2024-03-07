import numpy as np
from numpy.linalg import inv

# Load the training data
data_train = np.load('ContinuousTrain.npy', allow_pickle=True).item()
kin_train = data_train['kin']
rate_train = data_train['rate']

# Add a two bin lag between kinematics and firing rate
yTrain = kin_train[2:3103, :]
xTrain = np.hstack((rate_train[0:3101, :], np.ones((3101, 1))))  # Add vector of ones for baseline
f = inv(xTrain.T @ xTrain) @ xTrain.T @ yTrain  # Create linear filter

# Load the test data
data_test = np.load('ContinuousTest.npy', allow_pickle=True).item()
kin_test = data_test['kin']
rate_test = data_test['rate']

xTest = np.hstack((rate_test[0:3101, :], np.ones((3101, 1))))
yActual = kin_test[2:3103, :]
yFit = xTest @ f

# Calculate correlation coefficients
cc = np.corrcoef(yActual.T, yFit.T)[0, 1]

# Calculate root mean squared errors (has units of cm).
rmse = np.sqrt(np.mean((yActual - yFit) ** 2, axis=0))