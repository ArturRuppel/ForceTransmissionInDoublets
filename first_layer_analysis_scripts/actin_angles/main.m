%Author: Artur Ruppel

close all;
clear all;
clc;

nb_cells = 50;
fileimage = 'actin_ec.tif';
filemask = 'mask.mat';
folder = 'D:\2020_OPTO H2000 stimulate left half doublets and singlets\TFM_doublets\AR1to2';
%% store all paths
for i = 1:nb_cells    
    if i<10
       paths{i}=(cat(2, folder, '\cell0',num2str(i)));
    else
       paths{i}=(cat(2,folder, '\cell',num2str(i))); 
    end
end

%% load first frame of all actin images and segment
order_parameter = [];
angles_all = [];
for i = 1:nb_cells
        image=im2double(imread(fullfile(paths{i},fileimage),1));
        load(fullfile(paths{i},filemask));
        mask=mask(:,:,1); % we use only the first frame to segment the cell
        SE = strel('disk',10);
        SE2 = strel('disk',30);
%         mask = imerode(mask,SE);
        mask = imclose(mask,SE2);
        mask = imdilate(mask,SE);
        image_segmented = image.*mask;
        images_segmented_all(:,:,i) = image_segmented;
        [order_parameter_current, angles_current] = Actin_analysis(image_segmented)
        angles_all = [angles_all; angles_current];
        order_parameter = [order_parameter, order_parameter_current];
        %image_segmented_all(:,:,i) = image_segmented;
    disp(cat(2,'cell ',int2str(i)));
end
%%
% figure;
% ax = polaraxes;
% polarhistogram(abs(angles),60);
% thetalim([0 90])
% ax.ThetaDir = 'counterclockwise';
% ax.ThetaZeroLocation = 'right';

% %% plot all. angle data has to be loaded to matlab by hand
% clc
% close all
% edges = linspace(0,pi/2,30);
% 
% figure;
% ax = polaraxes;
% ax.ThetaDir = 'counterclockwise';
% ax.ThetaZeroLocation = 'right';
% 
% 
% subplot(1,3,1);
% polarhistogram(angles1to1,edges,'Normalization','probability');
% title('1to1');
% thetalim([0 90]);
% rlim([0 0.1]);
% 
% subplot(1,3,2);
% polarhistogram(angles1to2,edges,'Normalization','probability');
% title('1to2');
% thetalim([0 90]);
% rlim([0 0.1]);
% 
% subplot(1,3,3);
% polarhistogram(angles2to1,edges,'Normalization','probability');
% title('2to1');
% thetalim([0 90]);
% rlim([0 0.1]);
% % 
% % %%
% % angles1to1 = angles1to1 - pi/2;
% % angles1to2 = angles1to2 - pi/2;
% % angles2to1 = angles2to1 - pi/2;
% % angles1to1 = abs(angles1to1);
% % angles1to2 = abs(angles1to2);
% % angles2to1 = abs(angles2to1);