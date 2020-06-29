%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%code for classification on nuclei in 
% brecahad dataset using graph data
% derived in breacahad_graph_feature_extraction
% python script. graph data generated in 
% brecahad_feature_extraction and saved with
% export_graph_data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%graph data without node attributes
no_attributes_dir = dir("C:\Users\Brian\Desktop\MastersResearch\brecahad_graph_features\no_attributes");
no_attributes_dir = no_attributes_dir(3:end);

for i = 1:size(no_attributes_dir, 1)
   data = readmatrix(strcat(no_attributes_dir(i).folder, "\", no_attributes_dir(i).name)); 
   %remove last column, which is just nan
   data = data(:, 1:end-1);
   X = data(:,1:end-1);
   y = data(:, end);
   
   %force dataset to have equal class representation by randomly sampling 
   %   from the underrepresnted class. comment this code out to just use 
   %   whole dataset
   %   there are 16689 positive samples, and only 1460 negative 
   min_num_samples = size(find(~y),1);
   idx = [];
   %sample min_num_samples of positive samples
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%find a better way to do this
   while size(idx,1) < min_num_samples
       j = randi(size(y, 1));
       if y(j) == 1
           idx = [idx ; j];
       end
   end
   while size(idx,1) < 2*min_num_samples
       j = randi(size(y, 1));
       if y(j) == 0
           idx = [idx ; j];
       end
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   y = y(idx);
   X = X(idx, :);
   
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
   
   cnf = confusionmat(y_test, y_pred);
   
   acc = (cnf(1,1) + cnf(2,2)) / (sum(sum(cnf)));
   sen = cnf(2,2) / (cnf(2,2) + cnf(2,1));
   spec = cnf(1,1) / (cnf(1,1) + cnf(1,2));
   
   model_name = no_attributes_dir(i).name;
   model_name = strsplit(model_name, ".");
   model_name = model_name{1};
   fprintf("Model: %s\tAccuracy: %0.4f\tSensitivity: %0.4f\tSpecificity: %0.4f\n",...
       model_name, acc, sen, spec);
end