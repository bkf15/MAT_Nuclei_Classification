addpath(pwd + "/AOFSkeletons_Code");
fetch_samples = false;			%whether or not to grab and save the samples for training
num_epochs = 1000;				%number of epochs for cnn

labels = get_labels();
concordant_samples = zeros(1, 2);
for i = 1:size(labels.p1_labels, 1)
	if labels.p1_labels(i) == labels.p2_labels(i) && labels.p1_labels(i) == labels.p3_labels(i)
		concordant_samples = [concordant_samples; labels.p1_labels(i) labels.slide_names(i)];
	end
end

concordant_samples = concordant_samples(2:end, :);
%ignore 'other' samples
%fea_samples = concordant_samples(concordant_samples(:, 1) == 'Flat Epithelial', :);
%columnar_samples = concordant_samples(concordant_samples(:, 1) == 'Columnar', :);
adh_samples = concordant_samples(concordant_samples(:, 1) == 'ADH', :);
normal_samples = concordant_samples(concordant_samples(:, 1) == 'Normal Duct', :);

%get all the color images and move them to the training folders
if fetch_samples
	%get the samples for training
	bw_nuclei_ims = {};
	color_nuclei_ims = {};
	masked_color_nuclei_ims = {};
	%first, get normal samples
	for i = 1:size(adh_samples,1)		%size of adh because its smaller
		im_path = char(normal_samples{i,2});
		im_path = char(pwd + "/all_slides/" + im_path(1:strfind(im_path,'_')-1) + "/" + im_path);

		im = imread(im_path);

		nuclei_boundaries = PCA_nuclei_segmentation(im);

		[bw,color,masked] = get_nuclei_crops(nuclei_boundaries, im);
		bw_nuclei_ims = [bw_nuclei_ims, bw];
		color_nuclei_ims = [color_nuclei_ims, color];
		masked_color_nuclei_ims = [masked_color_nuclei_ims, masked];
	end
	%save normal sample images 
% 	for i = 1:size(bw_nuclei_ims, 2)
% 		imwrite(bw_nuclei_ims{i}, strcat("cnn_data/bw/normal/",string(i),".jpg"));
% 		imwrite(color_nuclei_ims{i}, strcat("cnn_data/color/normal/",string(i),".jpg"));
% 		imwrite(masked_color_nuclei_ims{i}, strcat("cnn_data/masked_color/normal/",string(i),".jpg"));
% 	end
% 	
	%generate normal skeletons and scores, save to training folder
% 	[bin_skeletons, scored_nuclei] = compute_nuclei_skeletons(bw_nuclei_ims);
% 	for i = 1:size(bin_skeletons, 2)
% 		imwrite(bin_skeletons{i}, strcat("cnn_data/binary_skel/normal/",string(i),".jpg"));
% 		imwrite(scored_nuclei{i}, strcat("cnn_data/weighted_nuclei_outline/normal/",string(i),".jpg"));
% 	end
% 	
	bw_nuclei_ims = {};
	color_nuclei_ims = {};
	masked_color_nuclei_ims = {};
	%now, adh samples
	for i = 1:size(adh_samples,1)
		im_path = char(adh_samples{i,2});
		im_path = char(pwd + "/all_slides/" + im_path(1:strfind(im_path,'_')-1) + "/" + im_path);

		im = imread(im_path);

		nuclei_boundaries = PCA_nuclei_segmentation(im);

		[bw,color,masked] = get_nuclei_crops(nuclei_boundaries, im);
		bw_nuclei_ims = [bw_nuclei_ims, bw];
		color_nuclei_ims = [color_nuclei_ims, color];
		masked_color_nuclei_ims = [masked_color_nuclei_ims, masked];
	end
	%save adh sample images 
	for i = 1:size(bw_nuclei_ims, 2)
		imwrite(bw_nuclei_ims{i}, strcat("cnn_data/bw/adh/",string(i),".jpg"));
		imwrite(color_nuclei_ims{i}, strcat("cnn_data/color/adh/",string(i),".jpg"));
		imwrite(masked_color_nuclei_ims{i}, strcat("cnn_data/masked_color/adh/",string(i),".jpg"));
	end

	%generate adh skeletons and scores, save to training folder
	[bin_skeletons, scored_nuclei] = compute_nuclei_skeletons(bw_nuclei_ims);
	for i = 1:size(bin_skeletons, 2)
		imwrite(bin_skeletons{i}, strcat("cnn_data/binary_skel/adh/",string(i),".jpg"));
		imwrite(scored_nuclei{i}, strcat("cnn_data/weighted_nuclei_outline/adh/",string(i),".jpg"));
	end

end

%now, reopen data as imagedatastore 
bw_imds = imageDatastore('cnn_data/bw','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
color_imds = imageDatastore('cnn_data/color','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
masked_color_imds = imageDatastore('cnn_data/masked_color','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
bin_skeleton_imds = imageDatastore('cnn_data/binary_skel','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
weighted_skeleton_imds = imageDatastore('cnn_data/weighted_nuclei_outline','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

size(bw_imds.Files)

%train and evaluate CNN on task of classifying normal vs. adh 

%CNN layers
%input layer 1D for BW case
layers_bw = [
    imageInputLayer([41 41 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%input layer 3D, or 3 channeled, for color cases
layers_color = [
    imageInputLayer([41 41 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

%number of files for training set
numTrainFiles = int16(size(bw_imds,1) * 0.75);
disp('Starting CNN training');

%first, do bw CNN
[imdsTrain,imdsValidation] = splitEachLabel(bw_imds,numTrainFiles,'randomize');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',num_epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers_bw,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp('Binary accuracy: ');
accuracy = sum(YPred == YValidation)/numel(YValidation)

%next, color CNN
[imdsTrain,imdsValidation] = splitEachLabel(color_imds,numTrainFiles,'randomize');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',num_epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers_color,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp('Full color accuracy: ');
accuracy = sum(YPred == YValidation)/numel(YValidation)


%finally, masked color CNN
[imdsTrain,imdsValidation] = splitEachLabel(masked_color_imds,numTrainFiles,'randomize');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',num_epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers_color,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp('Masked color image: ');
accuracy = sum(YPred == YValidation)/numel(YValidation)

%now, binary skeleton
[imdsTrain,imdsValidation] = splitEachLabel(bin_skeleton_imds,numTrainFiles,'randomize');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',num_epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers_bw,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp('Binary skeleton accuracy: ');
accuracy = sum(YPred == YValidation)/numel(YValidation)

%now, weighted skeleton
[imdsTrain,imdsValidation] = splitEachLabel(weighted_skeleton_imds,numTrainFiles,'randomize');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',num_epochs, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers_color,options);
YPred = classify(net,imdsValidation);
YValidation = imdsValidation.Labels;
disp('Scored skeleton accuracy: ');
accuracy = sum(YPred == YValidation)/numel(YValidation)