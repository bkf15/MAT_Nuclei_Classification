addpath(pwd + "/AOFSkeletons_Code");
addpath(pwd + "/find_skel_intersection");

%if data isn't generated or up to date, use 'classification_test.m' with
%	fetch_samples=true

N = 8;				%use the top N scoring skeleton points as the feature vector

%open binary nuclei boundary images
bw_imds = imageDatastore('cnn_data/bw','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
ribbon_features = [];			%N dimensional feature vector for ribbon symmetry 
taper_features = [];
separation_features = [];
combined_features = [];
ribbon_and_taper_features = [];
average_combined_features = [];

labels = [];					%labels(i)=label for image i
filenames = {};

%loop over all the images and generate the feature vectors 
for i = 1:size(bw_imds.Files, 1)
	nucleus = im2bw(imread(bw_imds.Files{i}));
    
    %align the nuclei along its major axis (dont think this is necessary)
    %nucleus = alignAlongMajorAxis(nucleus);
	
	%generate and thin the skeleton
	skel = generate_skeletons_from_img(nucleus, 'invert', true);
	skel = thin_skeleton(skel);
	
	[dists,~] = bwdist(nucleus);				%distance transform on nucleus
	skel_cc = bwconncomp(skel);				%skeleton CC
	
	%compute skeleton intersection pts
	intersect_pts = find_skel_intersection(skel);
    
	%compute skeleton endpoints 
	[end_y,end_x] = ind2sub(size(skel), find(bwmorph(skel, 'endpoints')));
	endpoints = [end_x,end_y];
    
    %compute symmetry score for skeleton points
	k = 4;									%smaller k, since we're dealing with small nuclei
	%remember: channel 1 = ribbon, 2 = taper, 3 = separation 
	skel_scores = compute_symmetry_for_sk_pts(dists, skel, k, skel_cc);
    
	%compute skeleton branch features (without any DN stuff) 
	[max_br_len, avg_br_len, num_br, bend] = ...
		compute_branch_features(skel, skel_scores, intersect_pts, endpoints, skel_cc, nucleus);
	continue;
    
    %compute rnn features on the scores
    
	labels = [labels, bw_imds.Labels(i)];			%add label to label list
	filenames = [filenames, bw_imds.Files{i}];
end

%analysis, find the top scoring images 
[~,idx] = sort(mean(ribbon_features,2)); % sort just the scores column
sortedrib = ribbon_features(idx,:);   % sort the whole matrix using the sort indices
sorted_labels = labels(idx);
sorted_filenames = filenames(idx);

%display the highest and lowest scoring images 
for i = 1:5
	figure; imshow(imread(sorted_filenames{i})); title(sprintf("Number %i highest scoring. Class: %s", i, sorted_labels(i)));
	figure; imshow(imread(sorted_filenames{size(sorted_filenames,1)-i})); title(sprintf("Number %i lowest scoring. Class: %s", i, sorted_labels(size(sorted_labels,1)-i)));
end

%RIBBON: create train test split 
cv = cvpartition(size(ribbon_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = ribbon_features(idx,:);
y_test = labels(idx);

X_train = ribbon_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('ribbon features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));

%TAPER: create train test split 
cv = cvpartition(size(taper_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = taper_features(idx,:);
y_test = labels(idx);

X_train = taper_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('taper features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));

%SEPARATION: create train test split 
cv = cvpartition(size(separation_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = separation_features(idx,:);
y_test = labels(idx);

X_train = separation_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);


%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('separation features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));

%COMBINED: create train test split 
cv = cvpartition(size(combined_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = combined_features(idx,:);
y_test = labels(idx);

X_train = combined_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('combined features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));

%RIBBON+TAPER: create train test split 
cv = cvpartition(size(ribbon_and_taper_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = ribbon_and_taper_features(idx,:);
y_test = labels(idx);

X_train = ribbon_and_taper_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('ribbon+taper features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));

%AVERAGE COMBINED: create train test split 
cv = cvpartition(size(average_combined_features,1),'HoldOut',0.15);
idx = cv.test;
X_test = average_combined_features(idx,:);
y_test = labels(idx);

X_train = average_combined_features(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,~] = fitclinear(X_train, y_train);
Mdl = fitcsvm(X_train,y_train,'KernelScale','auto','Standardize',true,...
    'OutlierFraction',0.1);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('average combined features with N=%d accuracy: %0.5f\n', N, (size(y_pred)-num_wrong) / size(y_pred));