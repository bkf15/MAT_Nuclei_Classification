%function to remove nuclei that don't reside in an epithelial region, as 
%	they are likely to be noisy (not useful for the diagnosis)
%	converts the image to HSV and uses thresholding 
function [nuclei_boundaries] = remove_non_epithelial_nuclei(nuclei_boundaries, im)
	hsv = rgb2hsv(im);
	
	%throw out nuceli if < thresh*100 % of the pixels are in an epithelial region 
	thresh = 0.9;
	
	%isolate value and threshold to get more purpleish regions
	hsv = hsv(:,:,3);
	hsv_temp = zeros(size(hsv));
	hsv_temp(hsv >= 0.50) = 0;
	hsv_temp(hsv < 0.50) = 1;
	hsv = hsv_temp;
	
	%post processing, remove small cc's, fill, dilate and smooth
	pp = bwareaopen(hsv, 200);
	pp = imfill(pp, 'holes');
	pp = bwmorph(pp, 'thicken', 10);
	pp = bwmorph(pp, 'close');
	pp = bwmorph(pp, 'majority');
	
	%figure; subplot(1,2,1); imshow(im); subplot(1,2,2); imshow(pp);
	
	%now, remove the nuclei that don't have at least 90% of their pixels 
	%	inside an epithelial region 
	cc = bwconncomp(nuclei_boundaries);
	for i = 1:cc.NumObjects
		%the pixels of the current nuclei
		pixels = cc.PixelIdxList{i};
		%if < thresh % of the nuclei pixels are in an epithelial region, 
		%	remove it
		if nnz(pp(pixels))/size(pixels,1) < thresh
			nuclei_boundaries(pixels) = 0;
		end
	end
end