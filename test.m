im = data(1).crop;

%generate and thin the skeleton
skel = generate_skeletons_from_img(im, 'invert', true);
skel = thin_skeleton(skel); 
%remove noise / skeletons generated outside the image
skel = clean_skeleton(skel, im);

%compute the scores
[dists,~] = bwdist(im);				%distance transform on nucleus
skel_cc = bwconncomp(skel);

%compute skeleton endpoints 
[end_y,end_x] = ind2sub(size(skel), find(bwmorph(skel, 'endpoints')));
endpoints = [end_x,end_y];

%compute skeleton intersection pts
intersect_pts = find_skel_intersection(skel);

%compute symmetry score for skeleton points
k = 4;									%smaller k, since we're dealing with small nuclei
%remember: channel 1 = ribbon, 2 = taper, 3 = separation 
skel_scores = compute_symmetry_for_sk_pts(dists, skel, k, skel_cc);

graph = generate_skel_graph(skel_scores, skel_cc);