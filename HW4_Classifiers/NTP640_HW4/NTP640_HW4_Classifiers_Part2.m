% These examples require MATLAB's Deep Learning Toolbox and the Image
% Processing Toolbox

% ------------------------------------------------------------------------
% Section 1: Create classifier network with predetermined Filters
% -------------------------------------------------------------------------
% Plot gratings of different orientations (stimuli)
orientations = linspace(-90, 90, 5);

n_col = length(orientations);
stimulus = grating(0); % Using the grating function to get one example for size
[h, w] = size(stimulus); % height and width of stimulus

% Creating a figure with multiple subplots
stimuli = cell(5,1);

f = figure;
f.Position(3:4) = [500 100];
for i = 1:n_col
    ori = orientations(i);
    subplot(1, n_col, i);
    stimuli{i} = grating(ori); % Generate grating for current orientation
    imagesc(stimuli{i})
    colormap gray; % Set colormap to gray for proper visualization
    title(sprintf('%.0f°', ori)); % Set title for each subplot
    axis off; % Hide axis for a cleaner look
end
sgtitle(sprintf('stimulus size: %d x %d', h, w)); % Set a super title for the whole figure

% -------------------------------------------------------------------------
% Create center-surround and Gabor filters

% Function creates two center-surround filters and Gabor Filters of
% different orientations to initialize convolution layer
Filters = createFilters(6,7);
Weights = permute(Filters, [2,3,4,1]);

%% ------------------------------------------------------------------------
% Set up classifier layers

% Our goal is to classify whether a particular grating was tilted to the 
% right (angle > 0) or not

C_in = 1;   % #input channels
C_out = 6;  % #convolution output channels
K = 7;      % Kernel size
Kpool = 8;  % Pooling size

layers = [
        imageInputLayer([h, w, C_in], 'Normalization', 'none')
        
        % Convolutional layer
        % Use random initial weights
        %convolution2dLayer(K, C_out, 'Padding','same') 
        % Use Filters as initial weights
        convolution2dLayer(K, C_out, 'Padding','same', 'Weights', ...
            Weights, 'Bias', zeros(1,1,6))
        
        % Pooling layer
        maxPooling2dLayer(Kpool, 'Stride', Kpool)
        
        % Fully connected layers
        fullyConnectedLayer(10)

        % To produce a categorical output
        fullyConnectedLayer(2)
        softmaxLayer
        classificationLayer
    ];

% Use matlab analyzer to display the network and inspect the number of
% parameters in each layer
analyzeNetwork(layers)

% Create 1000 input stimuli for training
n_train = 1000;
ori = (rand(1, n_train) - 0.5) * 180;
[input_stimuli, response] = stim_resp_from_ori(ori);


%% ------------------------------------------------------------------------
% Section 2: Train network, plot activations
% -------------------------------------------------------------------------
% train network

options = trainingOptions('sgdm', ...  % Stochastic Gradient Descent with Momentum
    'InitialLearnRate', 0.0005, ...
    'Momentum', 0.99, ...
    'MaxEpochs', 25, ...
    'MiniBatchSize', 100, ...
    'Shuffle', 'every-epoch', ...  % Shuffle training data before each training epoch
    'Verbose', true, ...  % Display training progress information
    'VerboseFrequency', 50);  % How often to display progress

net = trainNetwork(input_stimuli, response, layers, options);

% Plot trained filter weights
f = figure;
f.Position(3:4) = [600 100];

for i = 1:C_out
    subplot(1, C_out, i);
    imagesc(squeeze(net.Layers(2).Weights(:,:,:,i))); % Display the grating
    title(sprintf('Filter %.0f', i)); % Set title for each subplot
    colormap gray;
    axis off; % Hide axis for a cleaner look
end
%% Plot activations of different filters in convolution layer
for i = 1:length(orientations)
    act{i} = activations(net,stimuli{i},'conv');
end

im_a = [];
for i = 1:length(orientations)
    im_ai = cat(3, stimuli{i}, act{i});
    im_a = cat(1, im_a, im_ai);
end

I = imtile(im_a,'GridSize',[1 7],'ThumbnailSize',[10*48 2*64]);
figure
imshow(I)
title('original input image + activations for each of the 6 filters')


%% ------------------------------------------------------------------------
% Section 3: Compare to real neurons
% ------------------------------------------------------------------------ 
% Get activity from pool layer and FC layer and compare to real V1 data

V1 = load('Stringer2021_data_Neuromatch.mat');

[exp_stimuli, ~] = stim_resp_from_ori(V1.ori);


hidden_activity_pool = activations(net,exp_stimuli,'maxpool', ...
    'OutputAs', 'rows'); 
hidden_activity_FC = activations(net,exp_stimuli,'fc_1', ...
    'OutputAs', 'rows');

% Plot figure comparing the activity of real V1 neurons with the
% activations of neurons in the maxpool and first FC layer
% These are essentially tuning curves
figure
subplot(3,1,1)
hold on
plot(V1.ori, V1.resp_v1(:,randi(size(V1.resp_v1,2),3,1)))
xlim([-90,90])
title('Sample of real V1 neurons'); 

subplot(3,1,2)
hold on
plot(V1.ori, hidden_activity_pool(:,randi(size(hidden_activity_pool,2),3,1)))
xlim([-90,90])
title('Sample of neurons from pool layer'); 

subplot(3,1,3)
hold on
plot(V1.ori, hidden_activity_FC)
xlim([-90,90])
title('Neurons from fully connected layer'); 

%% Representational dissimilarity analysis 
RDM_V1 = computeRDM(V1.resp_v1);
RDM_pool = computeRDM(hidden_activity_pool);
RDM_FC = computeRDM(hidden_activity_FC);

f = figure;
f.Position(3:4) = [800 200];

subplot(1,3,1)
imagesc(RDM_V1)
clim([0,2])
title('V1 neurons'); 
axis square;
colorbar;

subplot(1,3,2)
imagesc(RDM_pool)
clim([0,2])
title('pool layer'); 
axis square;
colorbar;

subplot(1,3,3)
imagesc(RDM_FC)
clim([0,2])
title('fully connected layer');
axis square;
colorbar;

% Create a logical mask for the upper triangular part excluding the diagonal
mask = triu(true(size(RDM_V1)), 1);

% Extract off-diagonal elements using the mask
RDM_V1_offdiag = RDM_V1(mask);
RDM_pool_offdiag = RDM_pool(mask);
RDM_FC_offdiag = RDM_FC(mask);

% Compute the correlation coefficient between the off-diagonal elements
corrMatrixV1pool = corrcoef(RDM_V1_offdiag, RDM_pool_offdiag);
corrMatrixV1FC = corrcoef(RDM_V1_offdiag, RDM_FC_offdiag);
corr_coefs = [corrMatrixV1pool(1,2); corrMatrixV1FC(1,2)]  % Extract the correlation coefficient


function RDM = computeRDM(resp)
    % Compute the representational dissimilarity matrix (RDM)
    %
    % Args:
    %   resp (matrix): S x N matrix with population responses to
    %       each stimulus in each row
    %
    % Returns:
    %   matrix: S x S representational dissimilarity matrix
    % Efficient computation of 1 - correlation coefficient

    % Z-score responses to each stimulus
    zresp = zscore(resp, 0, 2);  % Normalize across columns (features)

    % Compute RDM
    RDM = 1 - (zresp * zresp') / size(zresp, 2);
end

%% ------------------------------------------------------------------------ 
% Helper functions
% ------------------------------------------------------------------------ 
% No need to change anything here, but feel free to look

function [input_stimuli, responses] = stim_resp_from_ori(ori)
    len_ori = length(ori);
    inputs = cell(len_ori,1);
    response = zeros(len_ori, 1);
    for i = 1:len_ori
        inputs{i} = grating(ori(i));
        % is it tilted to the right?
        response(i) = ori(i) > 0;
    end
    input_stimuli = cat(4, inputs{:});
    responses = categorical(response);
end

function filters = createFilters(out_channels, K)
    if nargin < 1
        out_channels = 6;
    end
    if nargin < 2
        K = 7;
    end

    % Make example filters, some center-surround and gabors
    grid = linspace(-K/2, K/2, K);
    [xx, yy] = meshgrid(grid, grid);
    
    % Create center-surround filters
    sigma = 1.1;
    gaussian = exp(-(sqrt(xx.^2 + yy.^2))/(2*sigma^2));
    wide_gaussian = exp(-(sqrt(xx.^2 + yy.^2))/(2*(sigma*2)^2));
    center_surround = gaussian - 0.5 * wide_gaussian;
    
    % Create gabor filters
    thetas = linspace(0, 180, out_channels-2+1);
    thetas = thetas(1:end-1) * pi/180;
    gabors = zeros(length(thetas), K, K);
    lam = 10;
    phi = pi/2;
    gaussian = exp(-(sqrt(xx.^2 + yy.^2))/(2*(sigma*0.4)^2));
    for i = 1:length(thetas)
        theta = thetas(i);
        x = xx*cos(theta) + yy*sin(theta);
        gabors(i, :, :) = gaussian .* cos(2*pi*x/lam + phi);
    end
    
    % Ensure center_surround and -1*center_surround are 3D
    center_surround_3D = reshape(center_surround, [1, K, K]);
    neg_center_surround_3D = reshape(-1*center_surround, [1, K, K]);
    
    % Concatenate the filters
    filters = cat(1, center_surround_3D, neg_center_surround_3D, gabors);
    
    % Normalize and mean center filters
    for i = 1:size(filters, 1)
        filterMax = max(max(abs(filters(i, :, :))));
        filters(i, :, :) = filters(i, :, :) / filterMax;
        filterMean = mean(mean(filters(i, :, :)));
        filters(i, :, :) = filters(i, :, :) - filterMean;
    end
end

function gratings = grating(angle, sf, res, patch)
    if nargin < 2
        sf = 1 / 28;
    end
    if nargin < 3
        res = 0.1;
    end
    if nargin < 4
        patch = false;
    end

    % Convert angle from degrees to radians
    angle = deg2rad(angle);

    % Define width and height of the image in pixels for res=1.0
    wpix = 640;
    hpix = 480;

    % Create meshgrid
    [xx, yy] = meshgrid(sf * (0:(wpix * res - 1)) / res, sf * (0:(hpix * res - 1)) / res);

    if patch
        % Create grating for localized patch
        gratings = cos(xx .* cos(angle + 0.1) + yy .* sin(angle + 0.1));
        gratings(gratings < 0) = 0;
        gratings(gratings > 0) = 1;
        xcent = size(gratings, 2) * 0.75;
        ycent = size(gratings, 1) / 2;
        [xxc, yyc] = meshgrid(1:size(gratings, 2), 1:size(gratings, 1));
        icirc = sqrt((xxc - xcent).^2 + (yyc - ycent).^2) < (wpix / 3 / 2 * res);
        gratings(~icirc) = 0.5;
    else
        % Create full-image grating
        gratings = cos(xx .* cos(angle) + yy .* sin(angle));
        gratings(gratings < 0) = 0;
        gratings(gratings > 0) = 1;
    end

    gratings = gratings - 0.5;
end
