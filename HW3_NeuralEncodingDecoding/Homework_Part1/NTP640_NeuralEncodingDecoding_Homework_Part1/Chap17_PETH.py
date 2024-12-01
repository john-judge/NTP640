import matplotlib.pyplot as plt
import numpy as np

def Chap17_PETH(neuronNum=None):
    # Load the NeuralData
    data = np.load('NeuralData.npy', allow_pickle=True).item()
    unit = data['unit']
    direction = data['direction']
    go = data['go']

    # Choose a default neuron if none specified
    if neuronNum is None:
        neuronNum = 129

    graph = [6, 3, 2, 1, 4, 7, 8, 9]  # Sets the order to plot graphs in
    fig, axs = plt.subplots(3, 3)

    # Set edges of the PET histogram
    edgesPeri = np.arange(-1, 1.02, 0.02)  # 20 ms bins
    psth = np.zeros((len(edgesPeri), 8))

    # Compute PETH for each direction
    for i in range(8):
        indDir = np.where(direction == i)[0]  # Find trials in a given direction
        numTrials = len(indDir)

        for j in range(numTrials):
            centerTime = go[indDir[j]]  # To center on start of movement time
            allTimes = unit[neuronNum]['times'] - centerTime  # Center spike times
            spikeTimes = allTimes[(allTimes > -1) & (allTimes < 1)]  # Pick 2 seconds around center time

            # Add histogram of present trial to PETH
            if len(spikeTimes) > 0:
                psth[:, i] += np.histogram(spikeTimes, bins=edgesPeri)[0]

        # Divide by the number of trials & bin size for a mean firing rate
        psthNorm = psth / numTrials / 0.02

    yMax = np.max(psthNorm)
    for i in range(8):
        ax = axs.flatten()[graph[i]-1]
        ax.bar(edgesPeri[:-1], psthNorm[:, i], width=0.02)
        ax.set_ylim([0, yMax])
        ax.set_xlim([-1.05, 1.05])

    h = axs[1, 1]
    h.axis('off')
    chanNum = unit[neuronNum]['chanNum']
    unitNum = unit[neuronNum]['unitNum']
    h.text(0.25, 0.5, f'Chan {chanNum}-{unitNum}')

    plt.tight_layout()
    plt.show()

# Usage:
Chap17_PETH()