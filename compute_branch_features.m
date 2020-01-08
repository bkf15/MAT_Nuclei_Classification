function [max_br_len, avg_br_len, num_br, bend] = compute_branch_features(skel, intersect_pts, endpoints, skel_cc, nuclei_im)
	%max and average branch length, number of branches 
	max_br_len = 0;
	avg_br_len = 0;
	num_br = 0;
	
	%bend: compute angle between furthest two endpoints, try to find how
	%'bent' the nuclei is 
	end1 = endpoints(1,:);						%find furthest 2 endpoints 
	end2 = endpoints(2,:);          %note that endpoints is (column, row)(x,y)
	max_d = pdist([end1;end2]);
	for(i = 3:size(endpoints, 1))
		dist1 = pdist([end1; endpoints(i,:)]);
		dist2 = pdist([end2; endpoints(i,:)]);
		if dist1>max_d && dist1>=dist2
			max_d = dist1;
			end2 = endpoints(i,:);
		elseif dist2>max_d && dist2>dist1
			max_d = dist2;
			end1 = endpoints(i,:);
		end
    end
	
    %get angle of line connecting the two furthest end points and the x
    %axis
    %note: mult m by -1 because y coords are technically negative 
    m = -(end2(2) - end1(2))/(end2(1) - end1(1));
    if m == -Inf
        m = Inf;
    end
    endpoint_angle = atan(m)*180 / pi;
    %get angle of the major axis of the nucleus
    props = regionprops(nuclei_im, 'Orientation', 'Centroid');
    %to avoid dealing with signs, add 90 to both angles
    endpoint_angle = endpoint_angle + 90;
    major_axis_angle = props.Orientation + 90;
    difference = max(endpoint_angle, major_axis_angle) - ...
        min(endpoint_angle, major_axis_angle)
    %SIGNS OF ANGLES NOT ALWAYS LINING UP!! check when difference is very
    %high
	bend = 0;
	
    %get number of branches by setting junctions = 0, then running
    %connected component analysis 
    if size(intersect_pts, 1) == 0
        num_br = 1;
        branches_pixels = skel_cc.PixelIdxList;
    else
        skel_branches = skel;
        skel_branches(intersect_pts(:,2), intersect_pts(:,1)) = 0;
        branches_cc = bwconncomp(skel_branches);
        num_br = branches_cc.NumObjects;
        branches_pixels = branches_cc.PixelIdxList;
    end
    %compute avg branch length
end