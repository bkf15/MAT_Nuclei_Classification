%function to compute the symmetry scores for the contours in an image 
%args:
%	dists - distance transform computed on the original binary image 
%	skeleton - binary image of the computed skeleton 
%	k - the symmetry score for a point Q_i will be computed over Q_i-k ...
%		Q_i+k
%	tau - threshold for ROC of maximaly inscribed discs between pts Q_i and
%		Q_i+1. see algorithm for details 
%	CC - contains connected component information for SKELETON
%returns an NxMx3 matrix, where the original slide image is NxM. scores has
%3 channels where channel 1 = ribbon symmetry, channel 2 = taper symmetry,
%channel 3= separation
function [scores] = compute_symmetry_for_sk_pts(dists, skeleton, k, CC)
	
	%init the symmetry image 
	skeleton_r_symmetry = zeros(size(skeleton));
	skeleton_t_symmetry = zeros(size(skeleton));
	skeleton_separation = zeros(size(skeleton));
	
	%loop over each connected component, trace it, compute symmetry 
	for i = 1:CC.NumObjects
		%loop over the pixels in the current CC
		for j = 1:(size(CC.PixelIdxList{i}))
			%store the distances to use to compute separation
			medial_radii = [];
			%trace 2*k points along current connected component, starting 
			% at the j'th point along the CC
			[start_row, start_col] = ind2sub(size(skeleton), CC.PixelIdxList{i}(j));
			%trace - try k steps in both directions, then
			% concatenate the points together
			trace_ccw = bwtraceboundary(skeleton, [start_row, start_col], ...
				 'S', 8, k+1,'counterclockwise');
			trace_cw = bwtraceboundary(skeleton, [start_row, start_col], ...
				 'S', 8, k+1,'clockwise');
			trace = [trace_ccw; trace_cw];
			trace = unique(trace, 'rows');	
			if size(trace, 1) == 0
				continue;
			end
			%find the index in trace of the point for which we are
			% assigning scores to (start_row, start_col)
			%this index will be the same as that in medial_radii
			p_ind = find(ismember(trace, [start_row, start_col], 'Rows'));
			%use distance transform for trace points to compute symmetry
			% score for the j'th skeleton point
			count_above_threshold = 0;
			for m = 1:size(trace, 1)-1
				medial_radii = [medial_radii dists(trace(m, 1), trace(m, 2))];
			end
			%add the last point in the trace's distance to the radii vec
			medial_radii = [medial_radii dists(trace(end, 1), trace(end, 2))];
			%compute symmetry score
			%add a small constant to prevent zero scores
			first_deriv = abs(gradient(medial_radii)) ./ max(abs(gradient(medial_radii)));
			ribbon_score = first_deriv(p_ind);
			second_deriv = abs(gradient(gradient(medial_radii))) ./ max(abs(gradient(gradient(medial_radii))));
			taper_score = second_deriv(p_ind);
			%store the computed score at the skeleton point
			skeleton_r_symmetry(start_row, start_col) = ribbon_score;
			skeleton_t_symmetry(start_row, start_col) = taper_score;
			%compute separation score
			sep_score = 1 - ((2 * k - 1) / trapz(medial_radii));
			if sep_score < 0
				sep_score = 0.1;
			end
			skeleton_separation(start_row, start_col) = sep_score;
		end
	end
	scores = cat(3, skeleton_r_symmetry, skeleton_t_symmetry, skeleton_separation);
end