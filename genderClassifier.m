% Define categories and root folder for training
categories = {'Male', 'Female'};
rootFolderTrain = 'genderDatasetTrain'; % Path to the training dataset

% Initialize imageDatastore for male and female categories
imds_train_male = imageDatastore(fullfile(rootFolderTrain, categories{1}), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg'); % Datastore for male images
imds_train_female = imageDatastore(fullfile(rootFolderTrain, categories{2}), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg'); % Datastore for female images

% Sort and limit the training images to 2000 for each category
numTrainImages = 2000;
imds_train_male.Files = sort(imds_train_male.Files); % Sort male images
imds_train_female.Files = sort(imds_train_female.Files); % Sort female images
imds_train_male.Files = imds_train_male.Files(1:min(numTrainImages, numel(imds_train_male.Files))); % Limit to 2000 images
imds_train_female.Files = imds_train_female.Files(1:min(numTrainImages, numel(imds_train_female.Files))); % Limit to 2000 images

% Combine the two imageDatastores for training
imds_train = imageDatastore([imds_train_male.Files; imds_train_female.Files], ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg'); % Combined training datastore

% Resize images to 32x32 for CNN input
inputSize = [32 32]; % Size for CNN input
imds_train_resized = augmentedImageDatastore(inputSize, imds_train); % Resized training images

% Define CNN Layers
% (Several convolution, batch normalization, ReLU, and max pooling layers, followed by fully connected, softmax, and classification layers)
layers = [
    imageInputLayer([32 32 3])
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(2) % For binary classification (male/female)
    softmaxLayer
    classificationLayer
];

% Define training options
opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 64, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress'); % SGD with momentum and specific training options

% Train the network
[net, info] = trainNetwork(imds_train_resized, layers, opts); % Training process

% Load and prepare test data (similar process as for training data)
% (Following similar steps as for training data but for testing)
rootFolderTest = 'genderDatasetTest';
imds_test_male = imageDatastore(fullfile(rootFolderTest, categories{1}), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg');
imds_test_female = imageDatastore(fullfile(rootFolderTest, categories{2}), ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg');

% Sort and limit the test images to 300 for each category
numTestImages = 300;
imds_test_male.Files = sort(imds_test_male.Files);
imds_test_female.Files = sort(imds_test_female.Files);
imds_test_male.Files = imds_test_male.Files(1:min(numTestImages, numel(imds_test_male.Files)));
imds_test_female.Files = imds_test_female.Files(1:min(numTestImages, numel(imds_test_female.Files)));

% Combine the two imageDatastores for testing
imds_test = imageDatastore([imds_test_male.Files; imds_test_female.Files], ...
    'LabelSource', 'foldernames', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', '.jpg');

% Resize images for testing
imds_test_resized = augmentedImageDatastore(inputSize, imds_test); % Resized test images

% Test the network
labels = classify(net, imds_test_resized); % Classify the test images
confMat = confusionmat(imds_test.Labels, labels); % Confusion matrix
confMatNormalized = confMat ./ sum(confMat, 2); % Normalize confusion matrix
accuracy = mean(diag(confMatNormalized)); % Calculate accuracy

% Display accuracy
disp('Accuracy:');
disp(accuracy);

% Display confusion matrix
figure;
confusionchart(confMat, categories); % Visualize confusion matrix
title('Confusion Matrix');

% Display a specific test image with its predicted label
figure;
imageIndex = 305; % Choose an index to display a specific test image

% Read and display the specified image
im = imread(imds_test.Files{imageIndex}); % Read the image at the chosen index
imshow(im); % Display the image

% Determine the color of the title based on the prediction accuracy
if labels(imageIndex) == imds_test.Labels(imageIndex)
    colorText = 'g'; % Green color for correct predictions
else
    colorText = 'r'; % Red color for incorrect predictions
end

% Display the title with the predicted label
title(char(labels(imageIndex)), 'Color', colorText); % Show the predicted label with color-coded accuracy

% Capture an image from the webcam
cam = webcam; % Initialize the webcam
pause(2); % Pause for 2 seconds to allow webcam to start
capturedImage = snapshot(cam); % Capture an image from the webcam
clear cam; % Release the webcam resource

% Display the captured image
figure;
imshow(capturedImage); % Show the captured image

% Preprocess the image (resize to the input size of the network)
inputSize = [32 32]; % Resize to match CNN input size
capturedImageResized = imresize(capturedImage, inputSize); % Resize the captured image

% Classify the image
imageForClassification = augmentedImageDatastore(inputSize, capturedImageResized); % Prepare the image for classification
predictedLabel = classify(net, imageForClassification); % Classify the image

% Display the predicted label on top of the image
colorText = 'b'; % Blue color for the displayed label
title(char(predictedLabel), 'Color', colorText); % Display the predicted label on the image
