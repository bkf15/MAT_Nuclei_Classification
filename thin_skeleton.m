%function to make it easier to iterate over skeletal points by 
%	thinning the skeleton to 1 pixel wide. there is probably a better
%	way to do this
function [thinned_skeleton] = thin_skeleton(skeleton)
	thinned_skeleton = bwmorph(skeleton, 'thin');
	%remove connected components with size < 8
	%thinned_skeleton = bwareaopen(thinned_skeleton, 8);
end