%code for exporting graph data for usage in python
load('brecahad_features.mat');
d = cell2table(cell(0,3), 'VariableNames', {'A', 'X', 'y'});
for i = 1:size(feat_data, 2)
    %adjacency matrix
    A = full(adjacency(feat_data(i).graph));
    %feature matrix
    X = feat_data(i).graph.Nodes;
    X = X{:,:};         %[Ribbon Taper Separation]
    %accumulate the data in a cell array
    t = cell2table({A, X, feat_data(i).label}, 'VariableNames', {'A', 'X', 'y'});
    d = [d ; t];
end
%convert to json and write out
txt = jsonencode(d);

fid = fopen('brecahad_nuclei_graph_data.json', 'w');
if fid == -1, error('Cannot create JSON file'); end
fwrite(fid, txt, 'char');
fclose(fid);