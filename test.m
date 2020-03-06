ims = dir('BreCaHAD_dataset/BreCaHAD/images');
ims = ims(3:end);

for i = 1:size(ims)
    im = imread(strcat(ims(i).folder, '\', ims(i).name));
    name = split(ims(i).name, '.');
    name = name{1};
    mkdir(strcat(pwd, '\Cell-Nuclei-Detection-and-Segmentation\brecahad_slides\', name));
    write_loc=strcat(pwd, '\Cell-Nuclei-Detection-and-Segmentation\brecahad_slides\', ...
        name, '\', name, '.tif');
    imwrite(im, write_loc);
end