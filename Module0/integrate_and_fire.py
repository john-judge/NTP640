import numpy as np
import matplotlib.pyplot as plt

# Model parameters
Rm = 10  # membrane resistance in MΩ
Cm = 1  # membrane capacitance in nF
tau_m = Rm * Cm  # membrane time constant in ms
V_th = -50  # spike threshold in mV
V_reset = -65  # reset potential in mV
V_rest = -65  # resting potential in mV

Current_source = 'step'  # 'step' / 'sine' - Sets the current source to be a step function or a sinusoidal function
refractory_period = False  # true / false - Determine if you want to simulate the refractory period of the neuron.

# Time parameters
dt = 0.1  # time step in ms
T = 200  # total time in ms
time = np.arange(0, T+dt, dt)  # time vector
N = len(time)  # number of time steps

# Initialize a variable to store the time stamps when a spike occurred.
spike_times = []
# Initialize membrane potential vector to resting potential
V = np.zeros(N)
V[0] = V_rest

# Current source
if Current_source == 'sine':
    # Time-varying input current: Sinusoidal current with some noise
    I = 2 * np.sin(2 * np.pi * 0.5 * time / 50) + 0.5 * np.random.randn(N)  # in nA
elif Current_source == 'step':
    # Step function
    I = 2*np.ones(N)
    I[:500] = 0
    I[-500:] = 0
else:
    raise ValueError('ERROR! Did not recognize the current source, use step or sine.')

# Simulate the Integrate-and-Fire model
if not refractory_period:
    print('Simulating the Integrate-and-Fire model without the refractory period')
    for t in range(1, N):
        dV = ((-V[t-1] + V_rest + I[t] * Rm) / tau_m) * dt
        V[t] = V[t-1] + dV

        if V[t] >= V_th:
            V[t] = V_reset
            spike_times.append(time[t])
else:
    print('Simulating the Integrate-and-Fire model with the refractory period')
    # Refractory period parameters
    t_ref = 5  # refractory period in ms
    last_spike_t = -t_ref  # time of the last spike

    for t in range(1, N):
        # Check if the neuron is in the refractory period
        if time[t] - last_spike_t < t_ref:
            V[t] = V_reset
        else:
            dV = ((-V[t-1] + V_rest + I[t] * Rm) / tau_m) * dt
            V[t] = V[t-1] + dV

            if V[t] >= V_th:
                V[t] = V_reset
                spike_times.append(time[t])
                last_spike_t = time[t]  # update the time of the last spike

# Data Visualization
fig, axs = plt.subplots(2, 1)
axs[0].plot(time, V)
axs[0].scatter(spike_times, [V_th]*len(spike_times), color='r')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane Potential (mV)')
axs[0].set_title('Membrane Potential and Spike Times')
axs[0].set_ylim([-90, -40])
axs[0].set_yticks(range(-90, -40, 10))

axs[1].plot(time, I)
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Input Current (nA)')
axs[1].set_title('Input Current')
axs[1].set_ylim([-3, 3])
axs[1].set_yticks(range(-3, 4))

plt.tight_layout()
plt.show()