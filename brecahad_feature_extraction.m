addpath(pwd + "/AOFSkeletons_Code");
addpath(pwd + "/find_skel_intersection");

case_jsons = dir('BreCaHAD_dataset/BreCaHAD/groundTruth');
case_jsons = case_jsons(3:end);     %elimiate . and ..

%original image loading, for the segmentation
%images = dir('BreCaHAD_dataset/BreCaHAD/images');
images = dir('Cell-Nuclei-Detection-and-Segmentation/brecahad_slides');

images = images(3:end);
%struct array containing nuclei crops and labels 
data = struct('crop', {}, 'label', {}, 'color_im', {});
%struct array containing the skeletons, scored and unscored, and the label
skeletons = struct('binary_skel', {}, 'scored_skel', {}, 'label', {});

missed_nuclei = 0;
caught_nuclei = 0;

%i'll be saving the data, so as not to have to get it each time 
for i = 1:size(case_jsons, 1)
    %check to make sure the gt labels align with images, when reading in
    %using dir (never actually happens, just gonna leave the check in here)
    im_name = images(i).name;
    case_name = case_jsons(i).name;
    if ~strcmp(im_name, case_name(1:end-5))
        fprintf('image %s not lining up with cases.', im_name);
        continue;
    end

    gt_file = fileread(strcat(case_jsons(i).folder, '\', case_jsons(i).name));
    ground_truth = jsondecode(gt_file); %struct with all x/y's of centroids of nuclei
    %note: need to convert centroid coords from [0,1] to actual coordinate 
    %im = imread(strcat(images(i).folder, '\', images(i).name));
    im = imread(strcat('Cell-Nuclei-Detection-and-Segmentation\brecahad_slides\', ...
        images(i).name, '\', images(i).name,'.tif'));
    
    %get nuclei boundaries
    %nuclei_boundaries = PCA_nuclei_segmentation(im, 800, 300);
    nuclei_boundaries =  imbinarize(imread(strcat('Cell-Nuclei-Detection-and-Segmentation\brecahad_slides\', ...
        images(i).name, '\mask.png')));
    %figure; imshow(imoverlay(im, nuclei_boundaries, 'red')); zoom on;
    
    %eliminate detected nuclei that don't contain one of the centroid
    %points in the dataset 
    nuclei_boundaries_filled = imfill(nuclei_boundaries, 'holes'); 
    labeled_filled_nuclei = bwlabel(nuclei_boundaries_filled);
    
    for i = 1:size(ground_truth.tumor, 1)
        %GT labels in range [0,1], need to convert to x y coords
        real_x = floor(ground_truth.tumor(i).x * size(im, 2));
        real_y = floor(ground_truth.tumor(i).y * size(im, 1));
        %t = insertMarker(im, [real_x real_y]);
        %figure; imshow(t);
        
        %check whether this coordinate is within a detected nuclei
        if nuclei_boundaries_filled(real_y, real_x) == 1
           %if so, grab the crop and label it 
           %need to ensure the crop has only the nuclei boundary we are
           %interested in, and no other boundaries 
           nuclei_num = labeled_filled_nuclei(real_y, real_x);
           t = labeled_filled_nuclei;
           t(t ~= nuclei_num) = 0;
           try
            cropped_nuclei = t(real_y - 40 : real_y + 40, ...
               real_x - 40 : real_x + 40);
            cropped_color = im(real_y - 40 : real_y + 40, ...
               real_x - 40 : real_x + 40, :);
           catch e
            %disp('unable to crop nuclei, too close to border');
            missed_nuclei = missed_nuclei + 1;
           end
           %now, convert back to the boundary image (from filled)
           cropped_nuclei = edge(cropped_nuclei, 'sobel');
           %last check, if theres more than 1 CC in the nuclei image
           cc = bwconncomp(cropped_nuclei);
           if cc.NumObjects > 1
                %disp('unable to crop nuclei, too close to border');
                missed_nuclei = missed_nuclei + 1;
                continue;
           end
           data(end+1) = struct('crop', cropped_nuclei, 'label' , 1, 'color_im', cropped_color);
           caught_nuclei = caught_nuclei + 1;
           
        %else, we didnt detect the ground truth nuclei
        else
            missed_nuclei = missed_nuclei + 1;
            %disp('failed to detect a ground truth tumor nuclei.');
        end
    end
    
    %now, do same for non-tumor 
    for i = 1:size(ground_truth.non_tumor, 1)
        %GT labels in range [0,1], need to convert to x y coords
        real_x = floor(ground_truth.non_tumor(i).x * size(im, 2));
        real_y = floor(ground_truth.non_tumor(i).y * size(im, 1));
        %check whether this coordinate is within a detected nuclei
        if nuclei_boundaries_filled(real_y, real_x) == 1
           %if so, grab the crop and label it 
           %need to ensure the crop has only the nuclei boundary we are
           %interested in, and no other boundaries 
           nuclei_num = labeled_filled_nuclei(real_y, real_x);
           t = labeled_filled_nuclei;
           t(t ~= nuclei_num) = 0;
           try
            cropped_nuclei = t(real_y - 40 : real_y + 40, ...
               real_x - 40 : real_x + 40);
            cropped_color = im(real_y - 40 : real_y + 40, ...
               real_x - 40 : real_x + 40, :);
           catch e
            %disp('unable to crop nuclei, too close to border');
            missed_nuclei = missed_nuclei + 1;
            continue;
           end
           %now, convert back to the boundary image (from filled)
           cropped_nuclei = edge(cropped_nuclei, 'sobel');
           %last check, if theres more than 1 CC in the nuclei image
           cc = bwconncomp(cropped_nuclei);
           if cc.NumObjects > 1
                %disp('unable to crop nuclei, too close to border');
                missed_nuclei = missed_nuclei + 1;
                continue;
           end
           data(end+1) = struct('crop', cropped_nuclei, 'label' , 0, 'color_im', cropped_color);
           caught_nuclei = caught_nuclei + 1;
           
        %else, we didnt detect the ground truth nuclei
        else
            missed_nuclei = missed_nuclei + 1;
            %disp('failed to detect a ground truth non-tumor nuclei.');
        end
    end
end
%fraction ~0.25, still around 4509 samples 
disp(strcat('Fraction of ground truth nuclei caught: ', num2str(caught_nuclei / (missed_nuclei+caught_nuclei))));


feat_data = struct('features', {}, 'label', {}, 'boundary', {}, 'skel_scores', {}, 'morph_feats', {}, 'color_im', {});
num_invalid_skeletons = 0;
invalid_inds = [];
%now, compute skeletons and scores for all the data samples
for i = 1:size(data, 2)
    %generate and thin the skeleton
	skel = generate_skeletons_from_img(data(i).crop, 'invert', true);
	skel = thin_skeleton(skel); 
    %remove noise / skeletons generated outside the image
    skel = clean_skeleton(skel, data(i).crop);
    if(skel == -1)
        num_invalid_skeletons = num_invalid_skeletons + 1;
        invalid_inds = [invalid_inds i];
        continue;           %invalid skeleton
    end
    
    %compute the scores
    [dists,~] = bwdist(data(i).crop);				%distance transform on nucleus
    skel_cc = bwconncomp(skel);
    
    %compute skeleton endpoints 
	[end_y,end_x] = ind2sub(size(skel), find(bwmorph(skel, 'endpoints')));
	endpoints = [end_x,end_y];
    
    %compute skeleton intersection pts
	intersect_pts = find_skel_intersection(skel);
    
    %compute symmetry score for skeleton points
	k = 4;									%smaller k, since we're dealing with small nuclei
	%remember: channel 1 = ribbon, 2 = taper, 3 = separation 
	skel_scores = compute_symmetry_for_sk_pts(dists, skel, k, skel_cc);
    
    %compute skeleton branch features (without any DN stuff) 
	feat = compute_branch_features(skel, skel_scores, intersect_pts, ...
                                    endpoints, skel_cc, data(i).crop);
   
    %compute morphological features, like eccentricity, major/minor axis
    morph_feats = get_morphological_features(data(i).crop);%look at classification_test for some of the features
    
    feat_data(end+1) = struct('features', feat, 'label', data(i).label, ...
        'boundary', data(i).crop, 'skel_scores', skel_scores, ...
        'morph_feats', morph_feats, 'color_im', data(i).color_im);
end
fprintf('Num invalid skeletons: %d', num_invalid_skeletons);

%save all the nuclei images and the features 
%save('BreCaHAD_dataset/my_features.mat', 'feat_data');
%if size(dir('BreCaHAD_dataset/my_data/no_blur/1'), 1) < 3
    for i = 1:size(feat_data, 2)
        %unblurred
        imwrite(feat_data(i).skel_scores, sprintf('BreCaHAD_dataset/my_data/no_blur/%d/%d.png',feat_data(i).label,i));
        %gaussian with sigma = 1
        imwrite(imgaussfilt(feat_data(i).skel_scores, 1), sprintf('BreCaHAD_dataset/my_data/1_blur/%d/%d.png',feat_data(i).label,i));
        %gaussian with sigma = 2
        imwrite(imgaussfilt(feat_data(i).skel_scores, 2), sprintf('BreCaHAD_dataset/my_data/2_blur/%d/%d.png',feat_data(i).label,i));
        %full color crop
        imwrite(feat_data(i).color_im, sprintf('BreCaHAD_dataset/my_data/color/%d/%d.png',feat_data(i).label,i));
    end
%end

%Now, extract features from alexnet 
net = alexnet;
inputSize = net.Layers(1).InputSize;
imds = imageDatastore('BreCaHAD_dataset/my_data/color','IncludeSubfolders',true,'LabelSource','foldernames');
%aimds = augmentedImageDatastore(inputSize(1:2),imds);
aimds = augmentedImageDatastore(inputSize,imds);

%get features from the last pooling layer 
%layer = 'pool5';        %Nx9216 
%layer = 'fc6';          %Nx4096
layer = 'fc8';          %Nx1000
%not sure which layer to extract features from 
color_cnn_features = activations(net, aimds, layer, 'OutputAs','rows');

imds = imageDatastore('BreCaHAD_dataset/my_data/no_blur','IncludeSubfolders',true,'LabelSource','foldernames');
aimds = augmentedImageDatastore(inputSize,imds);
skel_cnn_features = activations(net, aimds, layer, 'OutputAs','rows');

save('brecahad_features', 'color_cnn_features', 'skel_cnn_features', 'data', 'feat_data', 'imds', 'aimds', '-v7.3');