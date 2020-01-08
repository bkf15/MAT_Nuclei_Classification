function [nuclei_skeleton, nuclei_boundary_scores, lumen_boundary_scores] = compute_skeleton(impath)
	addpath(pwd + "/AOFSkeletons_Code");
	labels = get_labels();
	slide_im = imread(impath);
	%obtain the nuclei boundaries 
	nuclei_boundaries = PCA_nuclei_segmentation(slide_im); 
	
	%obtain the lumen boundaries
	%lumen_boundaries = lumen_segmentation2(slide_im, nuclei_boundaries);

	%call the function from the MAT library to compute the AOF skeleton
	nuclei_skeleton = generate_skeletons_from_img(nuclei_boundaries, 'invert', true);
	%lumen_skeleton = generate_skeletons_from_img(lumen_boundaries, 'invert', true);

	%thin each skeleton branch to 1 pixel 
	nuclei_skeleton = thin_skeleton(nuclei_skeleton);
	%lumen_skeleton = thin_skeleton(lumen_skeleton);
	
	%compute connected components of skeleton and boundaries
	CC_nuclei_skeleton = bwconncomp(nuclei_skeleton);
	CC_nuclei_boundaries = bwconncomp(nuclei_boundaries);
	%CC_lumen_skeleton = bwconncomp(lumen_skeleton);
	%CC_lumen_boundaries = bwconncomp(lumen_boundaries);
	
	%function to remove skeleton points not contained within a
	%nuclues/lumen
	nuclei_skeleton = remove_bad_skeleton_points(nuclei_skeleton, nuclei_boundaries, CC_nuclei_skeleton);
	%lumen_skeleton = remove_bad_skeleton_points(lumen_skeleton, lumen_boundaries, CC_lumen_skeleton);	
	
	%compute the distance transform on the binary nuclei boundary image
	[nuclei_dists, ~] = bwdist(nuclei_boundaries);
	%[lumen_dists, ~] = bwdist(lumen_boundaries);
	
	%2*k is the number of neighbors on each side to compute a points' symmetry
	% score over. IE for point Pi, compute over Pi-k...Pi+k
	k = 12;
	
	%compute local symmetry for skeleton points 
	%scores has 3 channels, channel 1 = ribbon, 2 =taper, 3 = separation
	nuclei_skeleton_scores = compute_symmetry_for_sk_pts(nuclei_dists, nuclei_skeleton, k, CC_nuclei_skeleton);
	%lumen_skeleton_scores = compute_symmetry_for_sk_pts(lumen_dists, lumen_skeleton, k, CC_lumen_skeleton);
	%from the above symmetry scores, transfer the score to each skeleton
	%point's contour points. scores has 3 channels, channel 1 = ribbon, 2 =
	%taper, 3 = separation
	nuclei_boundary_scores = compute_boundary_symmetry(nuclei_skeleton_scores(:,:,1), nuclei_skeleton_scores(:,:,2), nuclei_skeleton_scores(:,:,3), nuclei_skeleton, nuclei_boundaries, CC_nuclei_boundaries);
	%lumen_boundary_scores = compute_boundary_symmetry(lumen_skeleton_scores(:,:,1), lumen_skeleton_scores(:,:,2), lumen_skeleton_scores(:,:,3), lumen_skeleton, lumen_boundaries, CC_lumen_boundaries);
	%{
	%convert boundary_symmetry into rgb space so it can be overlaid on the
	%slide image
	map = hsv(256);
	rgbImage = ind2rgb(uint8(boundary_symmetry * 255), map); 
	rgbImage = uint8(rgbImage * 255);
	t = rgbImage(:, :, 1);
	t(t == 255) = 0;
	rgbImage(:, :, 1) = t;
	
	slide_im(rgbImage ~= 0) = 0;
	slide_im = slide_im + rgbImage;
	figure; imshow(slide_im); colormap(map); colorbar;
	
	%perform delauney triangulation
	scores = del_triangulation(CC_nuclei);
	
	%function to highlight nuclei with very high symmetry scores 
	%find_high_symmetry_nuclei(boundary_symmetry, slide_im, labels, impath);
	find_high_symmetry_nuclei(boundary_symmetry, slide_im, labels, impath);
	%}
end

