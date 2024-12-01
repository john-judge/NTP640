% These examples require MATLAB's Deep Learning Toolbox and the Image
% Processing Toolbox

% -------------------------------------------------------------------------
% Section 1: Set up and train a convolutional neural network on MINST dataset
% -------------------------------------------------------------------------

% Load MINST dataset in special MATLAB format
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

% Display some of the images in the datastore.

figure;
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end

% check size of image
img = readimage(imds,1);
size(img)

%% Training the network 
% Specify Training and Validation Sets
numTrainFiles = 750;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');

C_out = 8;  % #convolution output channels
K = 3;      % Kernel size

% Specify the convolutional neural network

layers = [
    imageInputLayer([28 28 1])      %LA: Why is the third input 1? (Grayscale)
    
    
    convolution2dLayer(K,C_out,'Padding','same')  
    % 3: filter size 3 x 3; 8: num of filters/feature maps; 
    % 'Padding' 'same': adjust padding so the layer has the same num 
    % neurons as input
    batchNormalizationLayer
    % normalizes activations and gradients --> speed up training
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(K,2*C_out,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(K,4*C_out,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10) %LA: Why does this layer have 10 neurons? (# classes in the target data, 10 digits)
    softmaxLayer    % Converts fullyConnectedLayer into probabilities
    classificationLayer];   %Output

analyzeNetwork(layers)

% Specify Training Options

% Train the neural network using stochastic gradient descent with momentum 
% (SGDM) with an initial learning rate of 0.01. Set the maximum number of 
% epochs to 4. An epoch is a full training cycle on the entire training 
% data set. Monitor the neural network accuracy during training by 
% specifying validation data and validation frequency. Shuffle the data 
% every epoch. The software trains the neural network on the training data 
% and calculates the accuracy on the validation data at regular intervals 
% during training. The validation data is not used to update the neural 
% network weights. Turn on the training progress plot, and turn off the 
% command window output.

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

% Train Network
net = trainNetwork(imdsTrain,layers,options);

% Classify Validation Images and Compute Accuracy
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)

%% ------------------------------------------------------------------------
% Section 2: Investigate the pretrained neural network
% -------------------------------------------------------------------------
f = figure;
f.Position(3:4) = [800 100];

for i = 1:C_out
    subplot(1, C_out, i);
    imagesc(squeeze(net.Layers(2).Weights(:,:,:,i))); % Display the grating
    title(sprintf('Filter %.0f', i)); % Set title for each subplot
    colormap gray;
    axis off; % Hide axis for a cleaner look
end

%% Select input image 
% Plot a figure of the images, the filtered images, and the output of the 
% first ReLu layer

im = readimage(imds, 9123);
act1 = activations(net,im,'conv_1');

% Each activation can take any value, so normalize the output using mat2gray.
% White pixels represent strong positive activations and black pixels 
% represent strong negative activations. A channel that is mostly gray does 
% not activate as strongly on the input image.

im8 = repmat(im, 1,1,8);
im_c1 = [mat2gray(im8); mat2gray(act1)];

% Many of the channels contain areas of activation that are both light and 
% dark. These are positive and negative activations, respectively. However, 
% only the positive activations are used because of the rectified linear 
% unit (ReLU). 

act_r1 = activations(net,im,'relu_1');
im_c1_r1 = [im_c1; mat2gray(act_r1)];

I = imtile(im_c1_r1,'GridSize',[1 8],'ThumbnailSize',[750 250]);
figure;
imshow(I)

%% Activations of second and third relu layers, and the fully connected layer
act_r2 = activations(net,im,'relu_2');
I = imtile(imresize(mat2gray(act_r2), size(im)),'GridSize',[2 8],'ThumbnailSize',[128 128]);
figure
imshow(I)

act_r3 = activations(net,im,'relu_3');
I = imtile(imresize(mat2gray(act_r3), size(im)),'GridSize',[4 8],'ThumbnailSize',[128 128]);
figure
imshow(I)

act_FC = activations(net,im,'fc', ...
    'OutputAs', 'rows');
f = figure;
f.Position(3:4) = [400, 100];
imagesc(act_FC)
axis off; % Hide axis for a cleaner look
colormap gray