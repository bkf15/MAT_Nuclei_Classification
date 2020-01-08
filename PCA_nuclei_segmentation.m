function [nuclei_boundaries] = PCA_segmentation(slide_im)
	%RGB_Im = imread([d f(ii).name]);
    %disp(f(ii).name);
	RGB_Im = slide_im;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    H = reshape(RGB_Im, [size(RGB_Im,1)*size(RGB_Im,2) 3]);
    %%% average color
    m = mean(H, 1);
    %%% Subtrach the mean from every pixel
    H = double(H) - double(repmat(m, [size(RGB_Im,1)*size(RGB_Im,2) 1]));
    % covariance matrix 
    C = (H'*H)/(size(RGB_Im,1)*size(RGB_Im,2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% Eigendecomposition of symmetric cov matrix 
    [V,D] = eig(C);
    D = diag(D);
    %%%%% re-order according to decaying eigenvalues %%%%%%%%%%%%%%%%%%%%%%
    [D,I] = sort(D);
    if D(1)>D(length(D))
        D = reverse(D); I = reverse(I);
    end
    V = V(:,I);
    
    %%%%%%%%%%%%%% projection onto the components %%%%%%%%%%%%%%%%%%%%%%%%%
    H = H*V;
    %%%%%%%%%%%%%%%%%%%%%%% new transformed image in PCA colorspace %%%%%%%
    U = reshape(H, [size(RGB_Im,1) size(RGB_Im,2) 3]);
    U1 = U(:,:,1);
    U2 = U(:,:,2);
    U3 = U(:,:,3);

%%% play with this threshold to get nuclei
    %U3(find(U3>-75))=0;
	U3(U3 > -100) = 0;
	U3 = U3 .* -1;
	%threshold to binary image
	bin_U3 = im2bw(U3, 0.5);
	%remove noise 
	bin_U3 = bwareaopen(bin_U3, 30);
	%fill in nuclei
	bin_U3 = imfill(bin_U3, 'holes');
	%erode then dilate the boundaries to smooth
	bin_U3 = bwmorph(bin_U3, 'open', 5);
	%remove noise 
	bin_U3 = bwareaopen(bin_U3, 30);
	%compute perim
	nuclei_boundaries = bwperim(bin_U3);
	%clean boundaries. any CC's with size > ø for some large ø are
	%probably not actually nuclei and should be removed 
	CC = bwconncomp(nuclei_boundaries);
	for i = 1:CC.NumObjects
		if size(CC.PixelIdxList{i}, 1) > 150
			nuclei_boundaries(CC.PixelIdxList{i}) = 0;
		end
	end
	%figure; imagesc(nuclei_boundaries);
	%figure; imshow(imoverlay(slide_im, nuclei_boundaries));