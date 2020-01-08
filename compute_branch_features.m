function [max_br_len, avg_br_len, num_br, bend] = compute_branch_features(skel, intersect_pts, endpoints, skel_cc, nuclei_im)
	%max and average branch length, number of branches 
	max_br_len = 0;
	avg_br_len = 0;
	num_br = 0;
	
	%bend: compute angle between furthest two endpoints, try to find how
	%'bent' the nuclei is 
	end1 = endpoints(1,:);						%find furthest 2 endpoints 
	end2 = endpoints(2,:);
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
	
	
	
	%major axis stuff
	%use regionprops to get vector describing major axis 
	props = regionprops(nuclei_im, 'Orientation', 'Centroid', 'MajorAxisLength');
	m = tand(props.Orientation(1)); %slope of major axis
	%we have slope, and we know Centroid is on the line, so plug in for b
	b = props.Centroid(1,2) - props.Centroid(1,1)*m;
	x = [0:40];
	y = (m*x) + b;		%line major axis falls on, not sure if this is necessary
	
	%formula for angle between two lines is:tan(ø)=|(m2-m1)/(1+m1m2)|
	
	bend = 0;
	%junction properties. angles, number of branches at junctions, may not
	%be useful since most images have no junctions 
end