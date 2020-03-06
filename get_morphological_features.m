function [feat] = get_morphological_features(bw_im)
    area = regionprops(bw_im, 'Area');
	eccentricity = regionprops(bw_im, 'Eccentricity');
	extent = regionprops(bw_im, 'Extent');
	max_axis = regionprops(bw_im, 'MajorAxisLength');
	min_axis = regionprops(bw_im, 'MinorAxisLength');
    conv_area = regionprops(bw_im, 'ConvexArea');
    circ = regionprops(bw_im, 'Circularity');
    eq_dia = regionprops(bw_im, 'EquivDiameter');
    fill_area = regionprops(bw_im, 'FilledArea');
    perim = regionprops(bw_im, 'Perimeter');
    solidity = regionprops(bw_im, 'Solidity');
    
    feat = struct('area', area, 'eccentricity', eccentricity, ...
        'extent', extent, 'max_axis', max_axis, 'min_axis', min_axis, ...
        'conv_area', conv_area, 'circ', circ, 'eq_dia', eq_dia, ...
        'fill_area', fill_area, 'perim', perim, 'solidity', solidity);
end