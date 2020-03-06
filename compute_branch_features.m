function [feat] = compute_branch_features(skel, ...
        skel_scores, intersect_pts, endpoints, skel_cc, nuclei_im)
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
        min(endpoint_angle, major_axis_angle);
    %SIGNS OF ANGLES NOT ALWAYS LINING UP!! check when difference is very
    %high
	bend = difference;
	
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
    avg_br_len = mean(cell2mat(cellfun(@length, branches_pixels, 'UniformOutput', false)));
    
    %find max length branch and its index
    [max_br_len, ind] = max(cell2mat(cellfun(@length, branches_pixels, 'UniformOutput', false)));
    
    %get the symmetry scores for skeleton points on largest branch
    ribbon = skel_scores(:,:,1);
    taper = skel_scores(:,:,2);
    separation = skel_scores(:,:,3);
    
    rib_scores = ribbon(branches_pixels{ind});
    taper_scores = taper(branches_pixels{ind});
    sep_scores = separation(branches_pixels{ind});
    
    %get the avg scores across the whole skeleton
    avg_ribbon = mean(ribbon(skel_cc.PixelIdxList{1}));
    avg_taper = mean(taper(skel_cc.PixelIdxList{1}));
    avg_sep = mean(separation(skel_cc.PixelIdxList{1}));
    
    %get other features, std, min, max of each score over the skel
    std_ribbon = std(ribbon(skel_cc.PixelIdxList{1}));
    std_taper = std(taper(skel_cc.PixelIdxList{1}));
    std_sep = std(separation(skel_cc.PixelIdxList{1}));
    min_ribbon = min(rib_scores);      %min/maxs on largest branch
    min_taper = min(taper_scores);
    min_sep = min(sep_scores);
    max_ribbon = max(rib_scores);      
    max_taper = max(taper_scores);
    max_sep = max(sep_scores);
    
    %other misc features, regionprops of the skeleton
    skel_props = regionprops(skel, 'Solidity', 'MajorAxisLength', 'MinorAxisLength', ...
        'Eccentricity');
    
    feat = struct('max_br_len', max_br_len, 'avg_br_len', avg_br_len, ...
        'num_br', num_br, 'bend', bend, 'rib_scores', rib_scores, ...
        'taper_scores', taper_scores, 'sep_scores', sep_scores, ...
        'avg_ribbon', avg_ribbon, 'avg_taper', avg_taper, 'avg_sep', ...
        avg_sep, 'skel_props', skel_props, 'std_ribbon', std_ribbon, ...
        'std_taper', std_taper, 'std_sep', std_sep, 'min_ribbon', min_ribbon, ...
        'min_taper', min_taper, 'min_sep', min_sep, 'max_ribbon', max_ribbon, ...
        'max_taper', max_taper, 'max_sep', max_sep);
end