function aligned_img = alignAlongMajorAxis(image)
    %rotate image so major axis is at 0 degrees
    props = regionprops(image, 'Orientation', 'Centroid');
    x = props.Centroid(1) + 10*cosd(props.Orientation);
    y = props.Centroid(2) - 10*sind(props.Orientation);
    %display original image and its major axis
    figure; imshow(image); hold on
    line([props.Centroid(1), x], [props.Centroid(2), y]);
    
    %rotate the image
    aligned_img = imrotate(image, props.Orientation, 'bilinear');
    %display rotated image and its major axis, which should be x axis now
    props = regionprops(aligned_img, 'Orientation', 'Centroid');
    x = props.Centroid(1) + 10*cosd(props.Orientation);
    y = props.Centroid(2) - 10*sind(props.Orientation);
    figure; imshow(aligned_img); hold on
    line([props.Centroid(1), x], [props.Centroid(2), y]);	
end