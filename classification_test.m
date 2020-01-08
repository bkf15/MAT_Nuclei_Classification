addpath(pwd + "/AOFSkeletons_Code");
fetch_samples = true;			%whether or not to grab and save the samples for training
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
fea_samples = concordant_samples(concordant_samples(:, 1) == 'Flat Epithelial', :);
columnar_samples = concordant_samples(concordant_samples(:, 1) == 'Columnar', :);
adh_samples = concordant_samples(concordant_samples(:, 1) == 'ADH', :);
normal_samples = concordant_samples(concordant_samples(:, 1) == 'Normal Duct', :);

%just crop/writeout the binary images
%get all the color images and move them to the training folders
before = 0;		%how many nuclei before removing non-epithelial
after = 0;		%how many after
if fetch_samples
	%get the samples for training
	bw_nuclei_ims = {};
	%first, get normal samples
	for i = 1:size(adh_samples,1)		%size of adh because its smaller
		im_path = char(normal_samples{i,2});
		im_path = char(pwd + "/all_slides/" + im_path(1:strfind(im_path,'_')-1) + "/" + im_path);

		im = imread(im_path);

		nuclei_boundaries = PCA_nuclei_segmentation(im);
		before = before + bwconncomp(nuclei_boundaries).NumObjects;
		
		%remove nuclei that aren't in epithelial regions 
		nuclei_boundaries = remove_non_epithelial_nuclei(nuclei_boundaries, im);
		after = after + bwconncomp(nuclei_boundaries).NumObjects;

		[bw,~,~] = get_nuclei_crops(nuclei_boundaries, im);
		bw_nuclei_ims = [bw_nuclei_ims, bw];
	end
	sprintf("num normal nuclei before: %d   num normal nuclei after: %d", before, after);
	
	%generate normal skeletons and scores, save to training folder
	[bin_skeletons, scored_nuclei] = compute_nuclei_skeletons(bw_nuclei_ims);
	for i = 1:size(bin_skeletons, 2)
		imwrite(bin_skeletons{i}, sprintf('cnn_data/binary_skel/adh/%d.jpg',i));
		%imwrite(scored_nuclei{i}, strcat("cnn_data/weighted_nuclei_outline/adh/",string(i),".jpg"));
	end
	
	%save normal sample images 
	for i = 1:size(bw_nuclei_ims, 2)
		imwrite(bw_nuclei_ims{i}, sprintf('cnn_data/bw/normal/%d.jpg', i));
	end
	
	bw_nuclei_ims = {};
	%now, adh samples
	for i = 1:size(adh_samples,1)
		im_path = char(adh_samples{i,2});
		im_path = char(pwd + "/all_slides/" + im_path(1:strfind(im_path,'_')-1) + "/" + im_path);

		im = imread(im_path);

		nuclei_boundaries = PCA_nuclei_segmentation(im);
		
		%remove nuclei that aren't in epithelial regions 
		nuclei_boundaries = remove_non_epithelial_nuclei(nuclei_boundaries, im);

		[bw,~,~] = get_nuclei_crops(nuclei_boundaries, im);
		bw_nuclei_ims = [bw_nuclei_ims, bw];
	end
	%save adh sample images 
	for i = 1:size(bw_nuclei_ims, 2)
		imwrite(bw_nuclei_ims{i}, sprintf('cnn_data/bw/adh/%d.jpg',i));
	end

	%generate adh skeletons and scores, save to training folder
	[bin_skeletons, scored_nuclei] = compute_nuclei_skeletons(bw_nuclei_ims);
	for i = 1:size(bin_skeletons, 2)
		imwrite(bin_skeletons{i}, sprintf('cnn_data/binary_skel/adh/%d.jpg',i));
		%imwrite(scored_nuclei{i}, strcat("cnn_data/weighted_nuclei_outline/adh/",string(i),".jpg"));
	end

end


%now, reopen data as imagedatastore 
bw_imds = imageDatastore('cnn_data/bw','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');
bin_skeleton_imds = imageDatastore('cnn_data/binary_skel','IncludeSubfolders',true,'FileExtensions','.jpg','LabelSource','foldernames');

%generate the naive shape descriptors 
bw_shape_descriptors = [];
labels = [];
for i = 1:size(bw_imds.Files, 1)
	bw_im = im2bw(imread(bw_imds.Files{i}));
	%see matlab regionprops page for descriptions of these 
	area = regionprops(bw_im, 'Area');
	eccentricity = regionprops(bw_im, 'Eccentricity');
	extent = regionprops(bw_im, 'Extent');
	max_axis = regionprops(bw_im, 'MajorAxisLength');
	min_axis = regionprops(bw_im, 'MinorAxisLength');
	
 	bw_shape_descriptors = [bw_shape_descriptors ; ...
 		eccentricity.Eccentricity extent.Extent max_axis.MajorAxisLength ...
 		min_axis.MinorAxisLength area.Area];

	labels = [labels ; bw_imds.Labels(i)];
end

%create train test split 
cv = cvpartition(size(bw_shape_descriptors,1),'HoldOut',0.3);
idx = cv.test;
X_test = bw_shape_descriptors(idx,:);
y_test = labels(idx);

X_train = bw_shape_descriptors(~idx, :);
y_train = labels(~idx);

%fit the SVM model
rng(1); % For reproducibility 
%[Mdl,FitInfo] = fitclinear(X_train, y_train);
Mdl = fitctree(X_train, y_train);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end

fprintf('naive feat accuracy: %0.5f\n', (size(y_pred)-num_wrong) / size(y_pred));
% 
% figure; imshow(imread(bw_imds.Files{max_eccen_ind})); title('max');
% figure; imshow(imread(bw_imds.Files{min_eccen_ind})); title('min');

%now do the test on the hierarchicalCentroid features
X = [];
labels = [];
for i = 1:size(bw_imds.Files, 1)
	bw_im = im2bw(imread(bw_imds.Files{i}));
	[vec, ~] = hierarchicalCentroid(bw_im,7,0);
	X = [X ; vec];
	labels = [labels ; bw_imds.Labels(i)];
end

cv = cvpartition(size(X,1),'HoldOut',0.3);
idx = cv.test;
X_test = X(idx,:);
y_test = labels(idx);

X_train = X(~idx, :);
y_train = labels(~idx);

%train and evaluate 
rng(1); % For reproducibility 
%[Mdl,FitInfo] = fitclinear(X_train, y_train);
Mdl = fitctree(X_train, y_train);

%predit on test set
y_pred = predict(Mdl,X_test);

%print out stats
num_wrong = 0;
for i = 1:size(y_pred)
	if y_pred(i) ~= y_test(i)
		num_wrong = num_wrong + 1;
	end
end
fprintf('centroid feat accuracy: %0.5f\n', (size(y_pred)-num_wrong) / size(y_pred));

