addpath(pwd + "/AOFSkeletons_Code");

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
%adh_samples = concordant_samples(concordant_samples(:, 1) == 'ADH', :);
normal_samples = concordant_samples(concordant_samples(:, 1) == 'Normal Duct', :);

for i = 1:size(normal_samples,1)
	im_path = char(normal_samples{i,2});
	im_path = char(pwd + "/all_slides/" + im_path(1:strfind(im_path,'_')-1) + "/" + im_path);
	
	im = imread(im_path);

	nuclei_boundaries = PCA_nuclei_segmentation(im);

	[bw_nuclei_ims,color_nuclei_ims,masked_color_nuclei_ims] = get_nuclei_crops(nuclei_boundaries, im);

	nuclei_skeletons = compute_nuclei_skeletons(bw_nuclei_ims);
end