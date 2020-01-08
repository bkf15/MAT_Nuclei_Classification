function [bin_skeletons, scored_nuclei] = compute_nuclei_skeletons(nuclei_ims)
	k = 5;				%how many pixels to iterate over for scores: 2k+1
	bin_skeletons = {};
	scored_nuclei = {};
	for i = 1:size(nuclei_ims,2)
		[nuclei_skeleton,dists] = generate_skeletons_from_img(nuclei_ims{i}, 'invert', true);
		nuclei_skeleton = thin_skeleton(nuclei_skeleton);
		%skeleton with distance scores 
		%dists(find(nuclei_skeleton==0)) = 0;
		%skeletons{end+1} = dists;
		bin_skeletons{end+1} = nuclei_skeleton;

		skeleton_cc = bwconncomp(nuclei_skeleton);
		boundary_cc = bwconncomp(nuclei_ims{i});
		%skeleton_sym_scores is 3 channel : channel 1 = ribbon, 2 = taper, 
		% 3 = separation
		skeleton_scores = compute_symmetry_for_sk_pts(dists, nuclei_skeleton, k, skeleton_cc);
		nuclei_scores = compute_boundary_symmetry(skeleton_scores(:,:,1), skeleton_scores(:,:,2), ...
			skeleton_scores(:,:,3), nuclei_skeleton, nuclei_ims{i}, boundary_cc);
		scored_nuclei{end+1} = nuclei_scores;
	end
end