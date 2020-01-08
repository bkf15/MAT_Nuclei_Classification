%return an array of cropped X by Y images centered at the nuclei in im 
function [bw_nuclei_crops,color_nuclei_crops,masked_color_nuclei_crops] = get_nuclei_crops(bw_im, color_im)
	cc = bwconncomp(bw_im);
	props = regionprops(cc,'centroid');
	centroids = uint16(cat(1, props.Centroid));
	bw_nuclei_crops = {};
	color_nuclei_crops = {};
	masked_color_nuclei_crops = {};
	for i = 1:cc.NumObjects
		t = zeros(size(bw_im));
		t(cc.PixelIdxList{i}) = 1;
		try
			bw_crop = t(centroids(i,2)-20:centroids(i,2)+20, centroids(i,1)-20:centroids(i,1)+20);
			color_crop = color_im(centroids(i,2)-20:centroids(i,2)+20, centroids(i,1)-20:centroids(i,1)+20,:);
			masked_color_crop = bsxfun(@times, color_crop, cast(imfill(bw_crop), 'like', color_crop));

			cc_crop = bwconncomp(bw_crop);
			%only save the crop if theres a single CC, IE the whole nuclei
			%	is contained in the crop
			if cc_crop.NumObjects == 1
				bw_nuclei_crops{end+1} = bw_crop;
				color_nuclei_crops{end+1} = color_crop;
				masked_color_nuclei_crops{end+1} = masked_color_crop;
			end
		end
	end
end