%CC contains the connected component information for NUCLEI
%scores is NxMx3, for NxM slide image, channel 1 = ribbon symmetry, channel
%2 = taper symmetry, channel 3 = separation
function [scores] = compute_boundary_symmetry(skeleton_r_symmetry, skeleton_t_symmetry, skeleton_separation, skeleton, boundaries, boundary_CC)
	boundary_r_symmetry = zeros(size(boundaries));
	boundary_t_symmetry = zeros(size(boundaries));
	boundary_separation = zeros(size(boundaries));
	%compute the distance transform on the skeleton image. thus, for some
	%point [i, j] on the boundary contour, we can check idx[i, j] to get
	%the linear coordinate of the associated skeleton point 
	[~, idx] = bwdist(skeleton);
	%loop over each connected component, trace it, compute symmetry 
	for i = 1:boundary_CC.NumObjects
		%loop over the pixels in the current CC
		for j = 1:(size(boundary_CC.PixelIdxList{i}))
			try
				%get row, col of the boundary pixel
				[bound_r, bound_c] = ind2sub(size(boundaries), boundary_CC.PixelIdxList{i}(j));
				%get closest skeleton point 
				[skel_r, skel_c] = ind2sub(size(boundaries), idx(bound_r, bound_c));
				%assign the boundary points symmetry score to be the max of 
				% its closest skeleton point, and its current score
				% (boundary points will have more than one associated
				% skeleton point, just not in the same region)
				boundary_r_symmetry(bound_r, bound_c) = ...
					max(max(skeleton_r_symmetry(skel_r, skel_c), boundary_r_symmetry(bound_r, bound_c)));
				boundary_separation(bound_r, bound_c) = ...
					max(max(skeleton_separation(skel_r, skel_c), boundary_separation(bound_r, bound_c)));
				boundary_t_symmetry(bound_r, bound_c) = ...
					max(max(skeleton_t_symmetry(skel_r, skel_c), boundary_t_symmetry(bound_r, bound_c)));
				if boundary_r_symmetry(bound_r, bound_c) == 0
					boundary_r_symmetry(bound_r, bound_c) = 0.1;
				elseif boundary_t_symmetry(bound_r, bound_c) == 0
					boundary_t_symmetry(bound_r, bound_c) = 0.1;
				elseif boundary_separation(bound_r, bound_c) == 0
					boundary_separation(bound_r, bound_c) = 0.1;
				end
			catch 
				warning('Error :(');
			end
		end
	end
	scores = cat(3, boundary_r_symmetry, boundary_t_symmetry, boundary_separation);
end