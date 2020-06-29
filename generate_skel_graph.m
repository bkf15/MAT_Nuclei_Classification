%create the graph of skeleton points. each point becomes a node. each node
%described by the 3 scores
function [g] = generate_skel_graph(scores, cc)
    %define all the nodes, one node for each non-zero value in the image
    %(each skel point). Will look like
    %   [Nodeid, row, col rib_score taper_score sep_score]
    nodes = [];
    for i = 1:size(cc.PixelIdxList{1}, 1)
       [r, c] =  ind2sub(size(scores(:,:,1)), cc.PixelIdxList{1}(i));
       nodes = [nodes ; i r c scores(r,c,1) scores(r,c,2) scores(r,c,3)];
    end
    
    %loop over the nodes and generate edges. edge bw two nodes if their
    %distance is <=sqrt(2). includes self loops. edges looks like:
    %   [s t w]         w is the weight
    edges = [];
    for i = 1:size(nodes, 1)
        for j = i:size(nodes, 1)
            if j~=i
                d = sqrt((nodes(i,2)-nodes(j,2))^2 + (nodes(i,3)-nodes(j,3))^2);
                if d <= sqrt(2)
                   edges = [edges ; nodes(i,1) nodes(j,1) 1];
                end
            end
        end
    end
    
    EdgeTable = table([edges(:,1:2)], edges(:,3), 'VariableNames', {'EndNodes', 'Weight'});
    NodeTable = table(nodes(:,4), nodes(:,5), nodes(:,6), ...
        'VariableNames', {'Ribbon score', 'Taper score', 'Separation score'});
    g = graph(EdgeTable, NodeTable);
    %figure; plot(g);
end