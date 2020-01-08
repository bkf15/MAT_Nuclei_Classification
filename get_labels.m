%returns a struct array with the following fields:
%	participant_i	- name of participant i
%	pi_labels		- labels given by participant i
%	slide_names		- file names of the slide images 
%	folder_name		- name of the folder the files are contained in
function [labels] = get_labels()
	%read all the annotations into a table. each row is a different sample. 
	%looking at row i, we should have: 
	%	annotations(i, 1) = participant 1 ('Jeff Fine')
	%	annotations(i, 2) = slide image file name (should match slides(i).name)
	%	annotations(i, 3) = participant 1 label
	%	annotations(i, 4) = folder name 
	%	annotations(i, 5) = participant 2 ('Dr. Carter')
	%	annotations(i, 6) = participant 2 label
	%	annotations(i, 7) = participant 3 ('Olga Navolotskaia')
	%	annotations(i, 8) = participant 3 label
	warning('OFF', 'MATLAB:table:ModifiedAndSavedVarnames')
	annotations = readtable(strcat(pwd, "/ADH_Subset1759_PathologistAnnotations.xls"));
	labels = struct;
	labels.folder = string(table2cell(annotations(1, 4)));
	labels.participant_1 = string(table2cell(annotations(1, 1)));
	labels.participant_2 = string(table2cell(annotations(1, 5)));
	labels.participant_3 = string(table2cell(annotations(1, 7)));
	labels.slide_names = string(table2cell(annotations(:, 2)));
	labels.p1_labels = string(table2cell(annotations(:, 3)));
	labels.p2_labels = string(table2cell(annotations(:, 6)));
	labels.p3_labels = string(table2cell(annotations(:, 8)));
end