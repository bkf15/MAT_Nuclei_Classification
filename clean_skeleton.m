%remove all but the largest connected component in the skeleton, as well
%   as any skeletal points not within the nucleus 
function [clean_skel] = clean_skeleton(skel, nucleus)
    clean_skel = skel;
    skel_cc = bwconncomp(skel);
    if skel_cc.NumObjects == 0
        clean_skel = -1;
        return;
    end
    %index of the largest CC in skel_cc.PixelIdxList
    largest_cc = find(cellfun('size', skel_cc.PixelIdxList, 1) == max(cellfun('size', skel_cc.PixelIdxList, 1)));
    largest_cc = largest_cc(1);     %in case of a tie, just take the first
    %CC indices for all but the largest cc
    s = setdiff([1:skel_cc.NumObjects], largest_cc);
    %set those pixels to 0
    for i = 1:size(s, 2)
        clean_skel(skel_cc.PixelIdxList{s(i)}) = 0;
    end
    
    %now, remove any noisy examples where the skeleton lies outside the
    %nucleus
    
    %sanity check, if theres still more than 1 CC, somethings wrng
    clean_skel_cc = bwconncomp(clean_skel);
    if clean_skel_cc.NumObjects > 1
       disp('too many connected components'); 
    end
end