%Function to remove the skeleton points that are not contained within
%	a nucleus. Since each nucleus is a closed region, there is a large
%	space between the nuclei that is open. computing the symmetry for these
%	skeleton points is unnecessary and costly
function [new_skeleton] = remove_bad_skeleton_points(skeleton, nuclei_boundaries, CC)
	%now fill in each nuclei with 1's. this makes determining whether a 
	%	skeleton point is contained within a nucleus trivial
	filled_nuclei = imfill(nuclei_boundaries, 'holes');
	%loop over each skeletal branch, and if is not contained within a
	%nuclei, get rid of it 
	new_skeleton = skeleton;
	for i = 1:CC.NumObjects
		%now, CC.PixelIdxList{i} contains linear indices for i'th skeletal
		% branch
		%convert linear indices to row/column 
		[r, c] = ind2sub(size(skeleton), CC.PixelIdxList{i});
		%note that either the whole skeleton is contained in a nuclei, or
		% none of it is. thus we only need to check one point, the first 
		if filled_nuclei(r(1), c(1)) == 0
			new_skeleton(CC.PixelIdxList{i}) = 0;
		end
		%else, just leave the branch as it was 
	end
	

	
