%select the best subset of features from the skeleton feature set 
load('brecahad_features.mat');

%first, skeletal feature data
y = vertcat(feat_data.label);
%initially, just grab all the features
X_all = [];
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
    X_all = [X_all ; max_br_len, avg_br_len, num_br, bend, avg_ribbon, ...
        avg_taper, avg_sep, std_ribbon, std_taper, std_sep, min_ribbon, ...
        min_taper, min_sep, max_ribbon, max_taper, max_sep, maj_axis, ...
        min_axis, eccen, solidity]; 
end


%step one:use feature extraction methods to get most relevant features
%MCMR algorithm (see https://www.mathworks.com/help/stats/feature-selection.html for more algorithms)
ranked_predictors = fscmrmr(X_all, y);
%try classification using the first i most important features
for i = 1:size(ranked_predictors, 2)
   X = X_all(:, ranked_predictors(1:i)); 
   
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

    %confusion matrix
    C = confusionmat(y_test, y_pred);

    %accuracy
    acc = size(find(y_pred == y_test), 1) / size(y_pred, 1);
    
    fprintf('Accuracy with %d top ranked features: %0.4f\n', i, acc);
end

fprintf('Now, trying all possible subsets...\n');

%generate all the possible subsets of features of size 3-num_feats-1
all_subsets = [];           
for i = 15:size(X_all, 2)-1
   subsets = nchoosek(1:size(X_all, 2), i); 
   %need to pad out the subset with 0's, so they're all the same size
   subsets = [subsets zeros(size(subsets, 1), size(X_all,2)-i)]; %will all be length 20
   all_subsets = [all_subsets ; subsets];
end
%over a million different subsets

%now, begin going through and finding the subset with the highest accuracy
best_accs = [0];         %store the accumulating accuracies
best_subsets = [0];      %and the corresponding subset indices 
best_cnfs = {0};
for i=1:size(all_subsets, 1)
    subset = all_subsets(i, :);
    subset = subset(find(subset));
    X = X_all(:, subset);
    
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

    %confusion matrix
    C = confusionmat(y_test, y_pred);

    %accuracy
    acc = size(find(y_pred == y_test), 1) / size(y_pred, 1);
    if acc > best_accs(end)
        best_accs = [best_accs acc];
        best_subsets = [best_subsets i];
        best_cnfs = [best_cnfs C];
        fprintf('New best accuracy: %0.4f found for subset:\n',acc);
        fprintf('%d ', subset);
        fprintf('\n');
    end
    
    if mod(i, 1000) == 0
       fprintf('On subset %d of %d\n', i, size(all_subsets, 1));
    end
end
best_cnfs = best_cnfs(2:end);
best_accs = best_accs(2:end);
best_subsets = best_subsets(2:end);

