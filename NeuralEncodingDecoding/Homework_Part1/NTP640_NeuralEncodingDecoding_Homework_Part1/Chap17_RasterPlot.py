import matplotlib.pyplot as plt
import numpy as np

def Chap17_RasterPlot(neuronNum=None):
    # Load the NeuralData
    data = np.load('NeuralData.npy', allow_pickle=True).item()
    unit = data['unit']
    direction = data['direction']
    instruction = data['instruction']

    # Choose a default neuron if none specified
    if neuronNum is None:
        neuronNum = 129

    graph = [6, 3, 2, 1, 4, 7, 8, 9]  # Sets the order to plot graphs in
    fig, axs = plt.subplots(3, 3)
    for i in range(8):
        indDir = np.where(direction == i)[0]  # Find trials in a given direction
        numTrials = len(indDir)
        ax = axs.flatten()[graph[i]-1]
        for j in range(numTrials):
            centerTime = instruction[indDir[j]]  # To center on instruction time
            allTimes = unit[neuronNum]['times'] - centerTime  # Center spike times
            spikeTimes = allTimes[(allTimes > -1) & (allTimes < 1)]  # Pick 2 seconds around center time

            # Plot a line
            for spikeTime in spikeTimes:
                ax.plot([spikeTime, spikeTime], [j-1, j], color='k')

        ax.set_ylim([0, numTrials])

    h = axs[1, 1]
    h.axis('off')
    chanNum = unit[neuronNum]['chanNum']
    unitNum = unit[neuronNum]['unitNum']
    h.text(0.25, 0.5, f'Chan {chanNum}-{unitNum}')

    plt.tight_layout()
    plt.show()

# Usage:
Chap17_RasterPlot()