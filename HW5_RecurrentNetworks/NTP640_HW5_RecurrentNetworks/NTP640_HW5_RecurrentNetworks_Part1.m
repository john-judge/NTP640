% -------------------------------------------------------------------------
% Section 1: Create Wilson-Cowan model
% -------------------------------------------------------------------------
% Default parameters

tau_E = 1.0;       % Timescale of the E population [ms]
a_E = 1.2;        % Gain of the E population
theta_E = 2.8;    % Threshold of the E population
tau_I = 2.0;      % Timescale of the I population [ms]
a_I = 1.0;        % Gain of the I population
theta_I = 4.0;    % Threshold of the I population

wEE = 9;        % E to E
wEI = 4.0;        % I to E
wIE = 13.0;       % E to I
wII = 11.0;       % I to I

T = 50.0;         % Total duration of simulation [ms]
dt = 0.1;         % Simulation time step [ms]

range_t = 0:dt:T;
Lt = length(range_t);

I_ext_E = 0.0;    % External input to E
I_ext_I = 0.0;    % External input to I
I_ext_E_T = I_ext_E * ones(1, Lt); % For entire duration 
I_ext_I_T = I_ext_I * ones(1, Lt);


%% ------------------------------------------------------------------------
% Define F-I curves for the E and I populations
% These are sigmoid functions minus a constant term.

% Define the range for x
x = 0:0.1:10;

% Display a_E and theta_E values
disp(['a_E: ', num2str(a_E), ', theta_E: ', num2str(theta_E)]);
disp(['a_I: ', num2str(a_I), ', theta_I: ', num2str(theta_I)]);

% Compute the F-I curve of the excitatory population
FI_exc = FIcurve(x, a_E, theta_E);

% Compute the F-I curve of the inhibitory population
FI_inh = FIcurve(x, a_I, theta_I);

% Visualize
figure; % Creates a new figure
plot_FI_EI(x, FI_exc, FI_inh); % Custom function to plot the F-I curves

%% ------------------------------------------------------------------------
% In Matlab functions have to be defined at the end of the script.
% This is the time to look at the simulate_wc function, which performs the
% numerical integration of the wilson and cowan model, given the many input
% parameters we define here and above.

% set initial values for rE and rI
rE_init1 = 0.32;    % Initial value of E
rI_init1 = 0.15;    % Initial value of I

% Simulate trajectories of the excitatory and inhibitory population
[rE1, rI1] = simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wEI, wIE, wII, I_ext_E_T, I_ext_I_T, rE_init1, rI_init1, dt, range_t);

% Simulate second trajectory
rE_init2 = 0.33;    % Initial value of E
rI_init2 = 0.15;    % Initial value of I

[rE2, rI2] = simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wEI, wIE, wII, I_ext_E_T, I_ext_I_T, rE_init2, rI_init2, dt, range_t);

my_test_plot(range_t, rE1, rI1, rE2, rI2)

%% ------------------------------------------------------------------------
% Section 2: Phase plane analysis of the WC model
% -------------------------------------------------------------------------
% Plot trajectories in the phase plane
figure;
hold on;
plot(rE1, rI1, 'b', 'DisplayName', 'Trajectory 1'); 
plot(rE_init1, rI_init1, 'ob'); 

xlabel('$r_E$', 'Interpreter', 'latex');
ylabel('$r_I$', 'Interpreter', 'latex');
legend('Location', 'best');


%% ------------------------------------------------------------------------
% Section 3: Compute and plot nullclines
% -------------------------------------------------------------------------

% Define ranges for rE and rI for nullcline computation
Exc_null_rE = linspace(-0.01, 0.96, 100);
Inh_null_rI = linspace(-0.01, 0.8, 100);

% For the E nullcline
Exc_null_rI = 1 ./ wEI .* (wEE .* Exc_null_rE - FI_inv(Exc_null_rE, a_E, theta_E) + I_ext_E);

% For the I nullcline
Inh_null_rE = 1 ./ wIE .* (wII .* Inh_null_rI + FI_inv(Inh_null_rI, a_I, theta_I) - I_ext_I);

% Visualize
figure;
hold on;
plot(Exc_null_rE, Exc_null_rI, 'b', 'DisplayName', 'E nullcline'); 
plot(Inh_null_rE, Inh_null_rI, 'r', 'DisplayName', 'I nullcline');

xlabel('$r_E$', 'Interpreter', 'latex');
ylabel('$r_I$', 'Interpreter', 'latex');
legend('Location', 'best');

% ------------------------------------------------------------------------
% add the vector field

% Define grid for vector field
EI_grid = linspace(0, 1, 20);
[rE, rI] = meshgrid(EI_grid, EI_grid);

% Calculate derivatives
drEdt = (-rE + FIcurve(wEE * rE - wEI * rI + I_ext_E, a_E, theta_E)) / tau_E;
drIdt = (-rI + FIcurve(wIE * rE - wII * rI + I_ext_I, a_I, theta_I)) / tau_I;

% Plot vector field
n_skip = 2; % Skipping indices for less cluttered vector field visualization
quiver(rE(1:n_skip:end, 1:n_skip:end), rI(1:n_skip:end, 1:n_skip:end), ...
       drEdt(1:n_skip:end, 1:n_skip:end), drIdt(1:n_skip:end, 1:n_skip:end), ...
       'AutoScaleFactor', 1.5, 'Color', [0.5 0.5 0.5]);

% Additional plotting details
xlabel('$r_E$', 'Interpreter', 'latex');
ylabel('$r_I$', 'Interpreter', 'latex');
title('Nullclines and Vector Field');


%% ------------------------------------------------------------------------
% Section 4: Short pulse induced persistent activity
% -------------------------------------------------------------------------
% Model noisy synaptic input according to an Ornstein-Uhlenbeck (OU) process

tau_ou = 1.0; % Tau for OU process in ms
sig_ou = 0.1; % Noise amplitude
myseed = 2020; % Random seed

% Prepare time vector
Lt = numel(range_t);

% Set random seed
rng(myseed);

% Initialize OU process
noise = randn(1, Lt); % Generate Gaussian noise
I_ou = zeros(1, Lt); % Initialize I_ou
I_ou(1) = noise(1) * sig_ou; % Initial condition

% Generate OU
for it = 1:Lt-1
    I_ou(it+1) = I_ou(it) + dt / tau_ou * (0 - I_ou(it)) + sqrt(2 * dt / tau_ou) * sig_ou * noise(it + 1);
end

% Add short pulse
SE = 0.0; % strength of pulse
t_start = 20; %start of pulse (ms)
t_lag = 10; % duration of pulse (ms)
N_start = round(t_start/dt);
N_lag = round(t_lag/dt);

I_ou(N_start:N_start+N_lag) = I_ou(N_start:N_start+N_lag) + SE;

rE_init3 = 0.1;    % Initial value of E
rI_init3 = 0.1;    % Initial value of I
[rE3, rI3] = simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wEI, wIE, wII, I_ou, I_ou, rE_init3, rI_init3, dt, range_t);

% Plotting
figure('Position', [100, 100, 800, 550]); % Set figure size
subplot(2,1,1)
    plot(range_t, I_ou, 'b');
    xlabel('Time (ms)');
    ylabel('$I_{\mathrm{OU}}$', 'Interpreter', 'latex');
    title('Ornstein-Uhlenbeck Process');
subplot(2,1,2)
    plot(range_t, rE3, 'b', 'DisplayName', 'E population'); % Plot rE2
    hold on; % Hold on to plot on the same axes
    plot(range_t, rI3, 'r', 'DisplayName', 'I population'); % Plot rI2
    xlabel('t (ms)'); % Label for X-axis
    ylabel('Activity'); % Label for Y-axis
    legend('show', 'Location', 'best'); % Show legend
    hold off; % Release the plot hold
    title('WC Model');


%% ------------------------------------------------------------------------ 
% Wilson-Cowan simulation function
% ------------------------------------------------------------------------ 

function [rE, rI] = simulate_wc(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wEI, wIE, wII, I_ext_E, I_ext_I, rE_init, rI_init, dt, range_t)
    % Simulate the Wilson-Cowan equations

    % Initialize activity arrays
    Lt = numel(range_t);
    rE = [rE_init, zeros(1, Lt - 1)];
    rI = [rI_init, zeros(1, Lt - 1)];
    
    % Simulate the Wilson-Cowan equations
    for k = 1:Lt-1
        % Calculate the derivative of the E population
        drE = dt / tau_E * (-rE(k) + FIcurve(wEE * rE(k) - wEI * rI(k) + I_ext_E(k), a_E, theta_E));

        % Calculate the derivative of the I population
        drI = dt / tau_I * (-rI(k) + FIcurve(wIE * rE(k) - wII * rI(k) + I_ext_I(k), a_I, theta_I));

        % Update using Euler's method
        rE(k + 1) = rE(k) + drE;
        rI(k + 1) = rI(k) + drI;
    end
end

%% ------------------------------------------------------------------------ 
% Helper functions
% ------------------------------------------------------------------------ 
% No need to change anything here, but feel free to look

function f = FIcurve(x, a, theta)
    % Population activation function, F-I curve 
    % x: input variable
    % a: gain or slope, controls how steep the curve is
    % theta: threshold parameter; shifts function left or right along
    % x-axis
    % subtraction of constant term ensures f is 0 at 0
    f = 1 ./ (1 + exp(-a .* (x - theta))) - 1 ./ (1 + exp(a * theta));
end

function f_inv = FI_inv(x, a, theta)
    % Inverse of the F-I curve defined above
    f_inv = -1 ./ a .* log((x + (1 + exp(a .* theta)).^-1).^-1 - 1) + theta;
end

% ------------------------------------------------------------------------ 
% plotting functions

function plot_FI_EI(x, FI_exc, FI_inh)
    % This function plots the F-I curves for excitatory and inhibitory populations
    plot(x, FI_exc, 'b-', 'LineWidth', 2); % Plot F-I curve for excitatory population in blue
    hold on; % Hold on to plot multiple datasets in the same figure
    plot(x, FI_inh, 'r-', 'LineWidth', 2); % Plot F-I curve for inhibitory population in red
    xlabel('Input');
    ylabel('Firing rate');
    legend('Excitatory', 'Inhibitory', 'Location','best');
    title('F-I curves for E and I populations');
    hold off;
end

function my_test_plot(t, rE1, rI1, rE2, rI2)
    % Function to plot the activities of E and I populations

    figure; % Create a new figure

    % First subplot
    ax1 = subplot(2, 1, 1); % Create a subplot in a 2x1 grid, first position
    plot(t, rE1, 'b', 'DisplayName', 'E population'); % Plot rE1
    hold on; % Hold on to plot on the same axes
    plot(t, rI1, 'r', 'DisplayName', 'I population'); % Plot rI1
    ylabel('Activity'); % Label for Y-axis
    legend('show', 'Location', 'best'); % Show legend
    hold off; % Release the plot hold

    % Second subplot
    ax2 = subplot(2, 1, 2); % Create a subplot in a 2x1 grid, second position
    plot(t, rE2, 'b', 'DisplayName', 'E population'); % Plot rE2
    hold on; % Hold on to plot on the same axes
    plot(t, rI2, 'r', 'DisplayName', 'I population'); % Plot rI2
    xlabel('t (ms)'); % Label for X-axis
    ylabel('Activity'); % Label for Y-axis
    legend('show', 'Location', 'best'); % Show legend
    hold off; % Release the plot hold

    linkaxes([ax1, ax2], 'xy'); % Link axes for both subplots
end