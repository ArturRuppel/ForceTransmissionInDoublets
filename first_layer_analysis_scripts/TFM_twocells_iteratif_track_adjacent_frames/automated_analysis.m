%Author: Artur Ruppel
%% initialize stock of data paths
clear all;
close all;
nb_cells = 1;
manualregistration = false;
beads=cell(2,nb_cells);
initial=cell(2,nb_cells);
brightfield=cell(2,nb_cells);
addpath(genpath('iteratif_2cells_film'));
addpath(genpath('visualisation'));

%% stock data paths
path=uigetdir('C:', 'Select folder containing data to be treated');
for i=1:nb_cells
    [beads_file, beads_path]=uigetfile(fullfile(path,'*.*'),cat(2,num2str(i),'. stack of beads to process'));
    beads{1,i}=beads_file;beads{2,i}=beads_path;
    
    [initial_file, initial_path]=uigetfile(fullfile(path,'*.*'),cat(2,num2str(i),'. initial state bead image'));
    initial{1,i}=initial_file;initial{2,i}=beads_path;
    
    [brightfield_file, brightfield_path]=uigetfile(fullfile(path,'*.*'),cat(2,num2str(i),'. stack of brightfield images (optional)'));
    brightfield{1,i}=brightfield_file;brightfield{2,i}=brightfield_path;  
end


[mask_left_file, mask_left_path]=uigetfile(fullfile(path,'*.*'), 'Stack of mask for cell');
[mask_right_file, mask_right_path]=uigetfile(fullfile(path,'*.*'),'Stack of mask for right cell');


% %% stock paths alternative
% for i=1:17
%     beads{1,i}=beads_file;
%     initial{1,i}=initial_file;
%     brightfield{1,i}=brightfield_file;
%     if i < 10
%         beads{2,i}          = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell0',num2str(i),'\');
%         initial{2,i}        = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell0',num2str(i),'\');
%         brightfield{2,i}    = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell0',num2str(i),'\'); 
%     else
%         beads{2,i}          = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell',num2str(i),'\');
%         initial{2,i}        = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell',num2str(i),'\');
%         brightfield{2,i}    = cat(2,'D:\2020_DRUGS H2000 Blebbistatin\AR2to1\cell',num2str(i),'\'); 
%     end
% end

% %% stock paths alternative
% for i=1:7
%     for j=1:7
%         beads{1,(i-1)*7+j}=beads_file;
%         initial{1,(i-1)*7+j}=initial_file;
%         brightfield{1,(i-1)*7+j}=brightfield_file;
%     
%         beads{2,(i-1)*7+j}          = cat(2,'D:\Data\20-06-02 - OPTO stimulate left cell X patterns - Y27\cell',num2str(i),'segment',num2str(j));
%         initial{2,(i-1)*7+j}        = cat(2,'D:\Data\20-06-02 - OPTO stimulate left cell X patterns - Y27\cell',num2str(i),'segment',num2str(j));
%         brightfield{2,(i-1)*7+j}    = cat(2,'D:\Data\20-06-02 - OPTO stimulate left cell X patterns - Y27\cell',num2str(i),'segment',num2str(j)); 
%     end
% end
%% run analyses
if manualregistration
    for i=1:nb_cells
        close all;
        track_film_TFM_iteratifPIV_interframe('BK+AK.tif',beads{2,i},0,initial{2,i}, brightfield{1,i}, brightfield{2,i}, mask_left_path, mask_left_file, mask_right_path, mask_right_file)
        make_movies(beads{2,i});
    end
else
    for i=1:nb_cells
        close all;
        track_film_TFM_iteratifPIV_interframe(beads{1,i},beads{2,i},initial{1,i},initial{2,i}, brightfield{1,i}, brightfield{2,i}, mask_left_path, mask_left_file, mask_right_path, mask_right_file)
        make_movies(beads{2,i});
    end
end


% 
% %% run visualisations
% for i=1:nb_cells
%     close all;
%     visualization_movies(beads{2,i});
% end
% 
% %% run FOM
% for i=1:nb_cells
%     close all;
%     FOM(beads{2,i});
% end