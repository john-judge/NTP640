% -------------------------------------------------------------------------
% Section 1: Create Wilson-Cowan model with three populations E1, E2, I
% -------------------------------------------------------------------------
% Default parameters

tau_E = 4.0;       % Timescale of the E population [ms]
a_E = 1.2;        % Gain of the E population
theta_E = 2.8;    % Threshold of the E population
tau_I = 1.5;      % Timescale of the I population [ms]
a_I = 1.0;        % Gain of the I population
theta_I = 4.0;    % Threshold of the I population

wEE = 4.0;        % E to E
wPlus = 6.0;      % stronger connectivity within population
wEI = 18.0;        % I to E
wIE = 6.0;       % E to I
wII = 6.0;       % I to I

T = 150.0;         % Total duration of simulation [ms]
dt = 0.1;         % Simulation time step [ms]

range_t = 0:dt:T;
Lt = length(range_t);

%% ------------------------------------------------------------------------
% Section 2: Plot example trajectory with or without noise
% -------------------------------------------------------------------------

I_ext_E1 = 2;    % External input to E1
I_ext_E2 = 2;    % External input to E2
I_ext_I = 1.;    % External input to I

% Add noise to inputs 
tau_ou = 1.0; % Tau for OU process in ms
sig_ou = 0.1; % Noise amplitude
myseed = 2050; % Random seed

% Prepare time vector
Lt = numel(range_t);

% Set random seed
rng(myseed);

I_ou = zeros(3, Lt); % Initialize I_ou
noise = randn(3, Lt); % Generate Gaussian noise 
I_ou(:,1) = noise(:,1) .* sig_ou; % Initial condition

% Generate OU
for it = 1:Lt-1
    I_ou(:,it+1) = I_ou(:,it) + dt / tau_ou .* (0 - I_ou(:,it)) + sqrt(2 * dt / tau_ou) * sig_ou .* noise(:,it + 1);
end

I_ext_E1_T = I_ou(1,:) + I_ext_E1; % For entire duration 
I_ext_E2_T = I_ou(2,:) + I_ext_E2; % For entire duration 
I_ext_I_T = I_ou(3,:) + I_ext_I;

% set initial values for rE and rI
rE1_init1 = 0.35;    % Initial value of E
rE2_init1 = 0.32;    % Initial value of E
rI_init1 = 0.15;    % Initial value of I

% Simulate trajectories of the excitatory and inhibitory population
[rE11, rE21, rI1] = simulate_wc2(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wPlus, wEI, wIE, wII, ...
                                    I_ext_E1_T, I_ext_E2_T, I_ext_I_T, rE1_init1, rE2_init1, rI_init1, dt, range_t);

my_test_plot(range_t, rE11, rE21, rI1)

%% ------------------------------------------------------------------------
% Section 3: Stability analysis
% -------------------------------------------------------------------------
% Feel free to explore what is going on below. No need to change anything
% in this section.

% Range of I_ext_E and initial condition range
I_ext_E_range = 0:0.2:10; % Example range
numInitialConditions = 150; % Number of initial guesses
tolerance = 1e-2; % Tolerance for considering two fixed points as distinct

% Preallocate storage for fixed points (using cells to accommodate variability in number of fixed points)
fixedPoints = cell(length(I_ext_E_range), 1);

options = optimoptions('fsolve', 'Display', 'off'); % Silence fsolve output for clarity

% Main loop over external input values
for i = 1:length(I_ext_E_range)
    
    uniqueFixedPoints = []; % Array to store unique fixed points for this input
    
    % Inner loop over multiple initial conditions
    for j = 1:numInitialConditions
        % Generate a random initial condition (or use a systematic approach)
        x0 = rand(1, 3); % Example with random initial condition
        
        % Define the Wilson-Cowan equations for this iteration
        % Same input to E1 and E2
        my_WC_two_E = @(x) [...
            (-x(1) + FIcurve((wEE + wPlus) * x(1) + wEE * x(2) - wEI * x(3) + I_ext_E_range(i), a_E, theta_E)) / tau_E; ...
            (-x(2) + FIcurve((wEE + wPlus) * x(2) + wEE * x(1) - wEI * x(3) + I_ext_E_range(i), a_E, theta_E)) / tau_E; ...
            (-x(3) + FIcurve(wIE * (x(1) + x(2)) - wII * x(3) + I_ext_I, a_I, theta_I)) / tau_I...
        ];
        
        % Attempt to find a fixed point
        [x_fp, ~, exitflag, ~] = fsolve(my_WC_two_E, x0, options);
        
        % Check if fsolve converged and if the solution is unique
        if exitflag > 0
            isUnique = true; % Assume the fixed point is unique initially
            for k = 1:length(uniqueFixedPoints)
                if norm(uniqueFixedPoints{k} - x_fp) < tolerance
                    isUnique = false; % Mark as not unique if close to any existing point
                    break; % Exit the loop early if a close point is found
                end
            end

            if isUnique
                uniqueFixedPoints{end+1} = x_fp; % Add the new unique fixed point
            end
        end
    end
    
    % Store the unique fixed points for this external input value
    % After finding a unique fixed point, check its stability
    for j = 1:length(uniqueFixedPoints)
        x_fp = uniqueFixedPoints{j};
        J = calculateJacobian(x_fp, wEE, wEI, wIE, wII, wPlus, a_E, theta_E, a_I, theta_I, tau_E, tau_I, I_ext_E_range(i), I_ext_E_range(i), I_ext_I);
        eigenvalues = eig(J);
        
        % Check stability
        if all(real(eigenvalues) < 0)
            % Stable fixed point
            stability = 'stable';
        else
            % Unstable fixed point
            stability = 'unstable';
        end
        
        % Store stability information along with fixed point
        uniqueFixedPoints{j} = struct('values', x_fp, 'stability', stability);
    end

    fixedPoints{i} = uniqueFixedPoints;

end

% Plot fixed point analysis for excitatory populations
figure; 
hold on;
for i = 1:length(I_ext_E_range)
    for j = 1:length(fixedPoints{i})
        % Plot with different markers or colors based on stability
        if strcmp(fixedPoints{i}{j}.stability, 'stable')
            plotStyle = 'k*'; % Example: black for stable
        else
            plotStyle = 'r.'; % Example: red for unstable
        end
        plot(I_ext_E_range(i), fixedPoints{i}{j}.values(1), plotStyle); % rE1
        plot(I_ext_E_range(i), fixedPoints{i}{j}.values(2), plotStyle); % rE2
    end
end
xlabel('I_{ext_E}');
ylabel('Fixed Points of Excitatory Populations');
title('Stability of Excitatory Fixed Points');
hold off;

% figure; % For the inhibitory population
% hold on;
% for i = 1:length(I_ext_E_range)
%     for j = 1:length(fixedPoints{i})
%         % Use the same stability marking as for the excitatory populations
%         if strcmp(fixedPoints{i}{j}.stability, 'stable')
%             plotStyle = 'k*'; % Example: black for stable
%         else
%             plotStyle = 'r.'; % Example: red for unstable
%         end
%         plot(I_ext_E_range(i), fixedPoints{i}{j}.values(3), plotStyle); % rI
%     end
% end
% xlabel('I_{ext_E}');
% ylabel('Fixed Points of Inhibitory Population');
% title('Stability of Inhibitory Fixed Points');
% legend({'Stable', 'Unstable'}, 'Location', 'best');
% hold off;


%% ------------------------------------------------------------------------ 
% Wilson-Cowan simulation function
% ------------------------------------------------------------------------ 

function [rE1, rE2, rI] = simulate_wc2(tau_E, a_E, theta_E, tau_I, a_I, theta_I, wEE, wPlus, wEI, wIE, wII, ...
                                    I_ext_E1, I_ext_E2, I_ext_I, rE1_init, rE2_init, rI_init, dt, range_t)
    % Simulate the Wilson-Cowan equations

    % Initialize activity arrays
    Lt = numel(range_t);
    rE1 = [rE1_init, zeros(1, Lt - 1)];
    rE2 = [rE2_init, zeros(1, Lt - 1)];
    rI = [rI_init, zeros(1, Lt - 1)];
    
    % Simulate the Wilson-Cowan equations
    for k = 1:Lt-1
        % Calculate the derivative of the E populations
        drE1 = dt / tau_E * (-rE1(k) + FIcurve((wEE+wPlus) * rE1(k) + wEE * rE2(k) - wEI * rI(k) + I_ext_E1(k), a_E, theta_E));
        drE2 = dt / tau_E * (-rE2(k) + FIcurve((wEE+wPlus) * rE2(k) + wEE * rE1(k) - wEI * rI(k) + I_ext_E2(k), a_E, theta_E));

        % Calculate the derivative of the I population
        drI = dt / tau_I * (-rI(k) + FIcurve(wIE * (rE1(k) + rE2(k)) - wII * rI(k) + I_ext_I(k), a_I, theta_I));

        % Update using Euler's method
        rE1(k + 1) = rE1(k) + drE1;
        rE2(k + 1) = rE2(k) + drE2;
        rI(k + 1) = rI(k) + drI;
    end
end

%% ------------------------------------------------------------------------ 
% Jacobian matrix for Wilson-Cowan model wrt excitatory input
% ------------------------------------------------------------------------ 
function J = calculateJacobian(x, wEE, wEI, wIE, wII, wPlus, a_E, theta_E, a_I, theta_I, tau_E, tau_I, I_ext_E1, I_ext_E2, I_ext_I)
    % Calculate derivatives of the FIcurve with respect to input for each population
    dF_dI_E1 = a_E * exp(-a_E * (((wEE + wPlus) * x(1) + wEE * x(2) - wEI * x(3) + I_ext_E1) - theta_E)) / (1 + exp(-a_E * (((wEE + wPlus) * x(1) + wEE * x(2) - wEI * x(3) + I_ext_E1) - theta_E)))^2;
    dF_dI_E2 = a_E * exp(-a_E * (((wEE + wPlus) * x(2) + wEE * x(1) - wEI * x(3) + I_ext_E2) - theta_E)) / (1 + exp(-a_E * (((wEE + wPlus) * x(2) + wEE * x(1) - wEI * x(3) + I_ext_E2) - theta_E)))^2;
    dF_dI_I = a_I * exp(-a_I * ((wIE * (x(1) + x(2)) - wII * x(3) + I_ext_I) - theta_I)) / (1 + exp(-a_I * ((wIE * (x(1) + x(2)) - wII * x(3) + I_ext_I) - theta_I)))^2;
    
    % Jacobian matrix components
    J11 = -1/tau_E + (wEE + wPlus) / tau_E * dF_dI_E1;
    J12 = wEE / tau_E * dF_dI_E1;
    J13 = -wEI / tau_E * dF_dI_E1;
    
    J21 = wEE / tau_E * dF_dI_E2;
    J22 = -1/tau_E + (wEE + wPlus) / tau_E * dF_dI_E2;
    J23 = -wEI / tau_E * dF_dI_E2;
    
    J31 = wIE / tau_I * dF_dI_I;
    J32 = wIE / tau_I * dF_dI_I;
    J33 = -1/tau_I + (-wII) / tau_I * dF_dI_I;
    
    % Assemble the Jacobian matrix
    J = [J11, J12, J13; J21, J22, J23; J31, J32, J33];
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

% ------------------------------------------------------------------------ 
% plotting functions
function my_test_plot(t, rE1, rE2, rI)
    % Function to plot the activities of E and I populations

    figure; % Create a new figure
    plot(t, rE1, 'b', 'DisplayName', 'E population'); % Plot rE1
    hold on; % Hold on to plot on the same axes
    plot(t, rE2, 'k', 'DisplayName', 'E population'); % Plot rE1
    plot(t, rI, 'r', 'DisplayName', 'I population'); % Plot rI1
    ylabel('Activity'); % Label for Y-axis
    legend('show', 'Location', 'best'); % Show legend
    hold off; % Release the plot hold
end