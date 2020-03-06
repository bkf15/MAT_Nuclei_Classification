%run brecahad_feature_extraction first to get the necessary variables 
%
load('brecahad_features.mat');

%first, skeletal feature data
y = vertcat(feat_data.label);
%extract all the features individually, to easier pick and choose
%note, doesnt seem that i can do multi-level struct indexing, so for loop
%it is
X = [];
X_alt = [];             %alternative feature definition
X_morph = [];           %morphological features
for i = 1:size(feat_data, 2)
    max_br_len = feat_data(i).features.max_br_len;
    avg_br_len = feat_data(i).features.avg_br_len;
    num_br = feat_data(i).features.num_br;
    bend = feat_data(i).features.bend;
    avg_ribbon = feat_data(i).features.avg_ribbon;
    avg_taper = feat_data(i).features.avg_taper;
    avg_sep = feat_data(i).features.avg_sep;
    std_ribbon = feat_data(i).features.std_ribbon;
    std_taper = feat_data(i).features.std_taper;
    std_sep = feat_data(i).features.std_sep;
    min_ribbon = feat_data(i).features.min_ribbon;
    min_taper = feat_data(i).features.min_taper;
    min_sep = feat_data(i).features.min_sep;
    max_ribbon = feat_data(i).features.max_ribbon;
    max_taper = feat_data(i).features.max_taper;
    max_sep = feat_data(i).features.max_sep;
    
    maj_axis = feat_data(i).features.skel_props.MajorAxisLength;
    min_axis = feat_data(i).features.skel_props.MinorAxisLength;
    eccen = feat_data(i).features.skel_props.Eccentricity;
    solidity = feat_data(i).features.skel_props.Solidity;
    %?? features in all
    X = [X ; max_br_len, avg_br_len, num_br, bend, avg_ribbon, ...
        avg_taper, avg_sep, std_ribbon, std_taper, std_sep, min_ribbon, ...
        min_taper, min_sep, max_ribbon, max_taper, max_sep, maj_axis, ...
        min_axis, eccen, solidity];
    
    X_morph = [ X_morph ; feat_data(i).morph_feats.area.Area, ...
        feat_data(i).morph_feats.eccentricity.Eccentricity, ...
        feat_data(i).morph_feats.extent.Extent, ...
        feat_data(i).morph_feats.max_axis.MajorAxisLength, ...
        feat_data(i).morph_feats.min_axis.MinorAxisLength, ...
        feat_data(i).morph_feats.conv_area.ConvexArea, ...
        feat_data(i).morph_feats.circ.Circularity, ...
        feat_data(i).morph_feats.eq_dia.EquivDiameter, ...
        feat_data(i).morph_feats.fill_area.FilledArea, ...
        feat_data(i).morph_feats.perim.Perimeter, ...
        feat_data(i).morph_feats.solidity.Solidity];
        
end

%create train test split 

%force dataset to have equal class representation by randomly sampling 
%   from the underrepresnted class. comment this code out to just use 
%   whole dataset
%   there are 16689 positive samples, and only 1460 negative 
min_num_samples = size(find(~y),1);
idx = [];
%sample min_num_samples of positive samples
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%find a better way to do this
while size(idx,1) < min_num_samples
    i = randi(size(y, 1));
    if y(i) == 1
        idx = [idx ; i];
    end
end
while size(idx,1) < 2*min_num_samples
    i = randi(size(y, 1));
    if y(i) == 0
        idx = [idx ; i];
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% y = y(idx);
% X = X(idx, :);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%perform the test 10 times, and store the accuracies
skel_accuracies = [];
%concatenate morph and skeleton features
X = [X X_morph];
for i = 1:10
    cv = cvpartition(size(X,1),'HoldOut',0.3);
    t_idx = cv.test;
    X_test = X(t_idx,:);
    y_test = y(t_idx);

    X_train = X(~t_idx, :);
    y_train = y(~t_idx);

    %fit the SVM model
    rng(1); % For reproducibility 
    %[Mdl,FitInfo] = fitclinear(X_train, y_train);
    Mdl = fitcsvm(X_train, y_train, 'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');

    %predit on test set
    y_pred = predict(Mdl,X_test);
    %accuracy
    acc = size(find(y_pred == y_test), 1) / size(y_pred, 1);
    skel_accuracies = [skel_accuracies acc];
end

%confusion matrix
C = confusionmat(y_test, y_pred);


fprintf('Explicit skeleton feature + morph accuracy: %0.4f\n', mean(skel_accuracies));
fprintf('STD of accuracies: %0.4f\n', std(skel_accuracies));
fprintf('Skeleton cnf matrix:\n[%d  %d\n%d  %d]\n', C);
return;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now, basic morph featers
X = X_morph;
morph_accuracies = [];
for i = 1:10
    cv = cvpartition(size(X,1),'HoldOut',0.3);
    t_idx = cv.test;
    X_test = X(t_idx,:);
    y_test = y(t_idx);

    X_train = X(~t_idx, :);
    y_train = y(~t_idx);

    %fit the SVM model
    rng(1); % For reproducibility 
    %[Mdl,FitInfo] = fitclinear(X_train, y_train);
    Mdl = fitcsvm(X_train, y_train, 'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');

    %predit on test set
    y_pred = predict(Mdl,X_test);
    
    %accuracy
    acc = size(find(y_pred == y_test), 1) / size(y_pred, 1);
    morph_accuracies = [morph_accuracies acc];
end
%confusion matrix
C = confusionmat(y_test, y_pred);


fprintf('nuclei morphology feature accuracy: %0.4f\n', mean(morph_accuracies));
fprintf('STD of accuracies: %0.4f\n', std(morph_accuracies));
fprintf('Morphology cnf matrix:\n[%d  %d\n%d  %d]\n', C);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%now, do CNN feature classification
%y = vertcat(imds.Labels);
X = color_cnn_features; 
%X = X(idx, :);
cnn_accuracies = [];
for i = 1:10
    %create train test split 
    cv = cvpartition(size(X,1),'HoldOut',0.3);
    idx = cv.test;
    X_test = X(idx,:);
    y_test = y(idx);

    X_train = X(~idx, :);
    y_train = y(~idx);

    %fit the SVM model
    rng(1); % For reproducibility 
    %[Mdl,FitInfo] = fitclinear(X_train, y_train);
    Mdl = fitcsvm(X_train, y_train, 'Standardize',true,'KernelFunction','RBF',...
        'KernelScale','auto');

    %predit on test set
    y_pred = predict(Mdl,X_test);

    %accuracy
    acc = size(find(y_pred == y_test), 1) / size(y_pred, 1);
    cnn_accuracies = [cnn_accuracies acc];
end
%confusion matrix
C = confusionmat(y_test, y_pred);


fprintf('CNN color feature accuracy: %0.4f\n', mean(cnn_accuracies));
fprintf('STD of accuracies: %0.4f\n', std(cnn_accuracies));
fprintf('CNN cnf matrix:\n[%d  %d\n%d  %d]\n', C);
