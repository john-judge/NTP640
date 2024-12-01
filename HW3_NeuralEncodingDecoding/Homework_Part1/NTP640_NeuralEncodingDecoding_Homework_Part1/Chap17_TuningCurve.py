import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Chap17_RasterPlot(neuronNum=None):
    # Load the NeuralData
    data = np.load('NeuralData.npy', allow_pickle=True).item()
    unit = data['unit']
    direction = data['direction']
    instruction = data['instruction']

    # Choose a default neuron if none specified
    if neuronNum is None:
        neuronNum = 129

    # Bin firing rates for each direction
    spikeCount = np.zeros(8)
    numTrials = np.zeros(8)

    for i in range(8):
        indDir = np.where(direction == i)[0]  # Find trials in a given direction
        numTrials[i] = len(indDir)

        for j in range(numTrials[i]):
            centerTime = instruction[indDir[j]]  # To center on instruction time
            allTimes = unit[neuronNum]['times'] - centerTime  # Center spike times
            spikeCount[i] += np.sum((allTimes > -0.5) & (allTimes < 1))  # Pick 2 seconds around center time

        # Divide by the number of trials & bin size (2 s) for a mean firing rate
        spikeCount[i] /= numTrials[i] * 1.5

    # Fit a tuning curve to "spikeCount"
    ang = np.arange(0, 360, 45)
    myfun = lambda theta, p1, p2, p3: p1 + p2 * np.cos(np.radians(theta) - p3)
    param, _ = curve_fit(myfun, ang, spikeCount, p0=[1, 1, 0])
    fit = myfun(np.arange(0, 360), *param)

    # Plot raw data, tuning curve
    plt.figure()
    plt.plot(ang, spikeCount)
    plt.xlabel('Angle')
    plt.ylabel('Avg Firing Rate')
    chanNum = unit[neuronNum]['chanNum']
    unitNum = unit[neuronNum]['unitNum']
    plt.title(f'Chan {chanNum}-{unitNum}')
    plt.plot(np.arange(0, 360), fit, 'r-.')
    plt.legend(['Actual', 'Fit'])

    # Easiest to pick preferred direction (in degrees) from fit data
    prefDir = np.argmax(fit)
    print(f'Preferred Direction: {prefDir}')

    corrcoef = np.corrcoef(myfun(ang, *param), spikeCount)[0, 1]
    print(f'Correlation Coefficient: {corrcoef}')