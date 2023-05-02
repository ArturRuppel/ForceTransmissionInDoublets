%Author: Artur Ruppel

close all;
clear all;
clc;

nb_cells = 17;
filename = 'actin_ec.tif';
%% store all paths
for i = 1:nb_cells    
    if i<10
       paths{i}=(cat(2,'D:\2021_OPTO H2000 stimulate all for 10 minutes\singlets\cell0',num2str(i)));
    else
       paths{i}=(cat(2,'D:\2021_OPTO H2000 stimulate all for 10 minutes\singlets\cell',num2str(i))); 
    end
end

%% run fiber tracking
for i = 16:17
    close all;
    info=imfinfo(fullfile(paths{i},filename),'tif');
    N_images=numel(info);
    disp([num2str(N_images),' images found.']);

    image=imread(fullfile(paths{i},filename),'tif');
    [M,N]=size(image);
    I=uint16(zeros(M,N,N_images));

    for frame=1:N_images
        I(:,:,frame)=imread(fullfile(paths{i},filename),frame);
    end

    I_rot = rot90(I); % fiber tracking algorithm only tracks top and bottom fiber, so I rotate the image and feed both the normal and the rotated image to the algorithm
    corners = readPoints(I(:,:,1),8);
    center = round(size(I(:,:,1))/2)';

    Rccw = [0 1; -1 0]; % Matrix to rotate cornerpoints left and right 90° ccw
    Rcw = [0 -1; 1 0];  % Matrix to rotate resulting points back, 90° cw

    corners_topbottom = corners(:,[1,2,5,6]);
    corners_leftright = corners(:,[3,4,7,8]);
    corners_leftright_rot= Rccw*(corners_leftright-center)+center;

    [Xtop, Ytop, Xbottom, Ybottom] = actin_contour_detection(I,corners_topbottom);
    %[Xright, Yright, Xleft, Yleft] = actin_contour_detection(I_rot,corners_leftright_rot);
    for j = 1:N_images
        Xright(:,j) = linspace(corners(1,3),corners(1,4),20)';
        Yright(:,j) = linspace(corners(2,3),corners(2,4),20)';
        Xleft(:,j) = linspace(corners(1,7),corners(1,8),20)';
        Yleft(:,j) = linspace(corners(2,7),corners(2,8),20)';    
    end
    
%     points_left = zeros([size(Xleft) 2]);
%     points_right = zeros([size(Xright) 2]);
%     for j = 1:N_images
%         points_left(:,j,:) = (Rcw*([Xleft(:,j), Yleft(:,j)]'-center) + center)';
%         points_right(:,j,:) = (Rcw*([Xright(:,j), Yright(:,j)]'-center) + center)';
%     end
%     Xleft = squeeze(points_left(:,:,1));
%     Yleft = squeeze(points_left(:,:,2));
%     Xright = squeeze(points_right(:,:,1));
%     Yright = squeeze(points_right(:,:,2));
    
    save(fullfile(paths{i},'fibertracking.mat'),'Xtop','Ytop','Xright','Yright','Xbottom','Ybottom','Xleft','Yleft');
    
%     h1 = figure('units','Normalized','position',[0.02 0.05 0.4 0.4],'Name','Fiber Tracking');
%     for j = 1:N_images % make movies        
%         figure(h1),
%         imshow(I(:,:,j)); hold on;
%         plot(Xtop(:,j),Ytop(:,1),'LineWidth',3,'Color','Red');
%         plot(Xbottom(:,j),Ybottom(:,j),'LineWidth',3,'Color','Red');
%         plot(Xleft(:,j),Yleft(:,j),'LineWidth',3,'Color','Red');
%         plot(Xright(:,j),Yright(:,j),'LineWidth',3,'Color','Red');
%
%         if ~exist(fullfile(paths{i},'fibertracking'),'dir')
%             mkdir(fullfile(paths{i},'fibertracking'))
%         end
% 
%         figurepath=cat(2,paths{i},'\fibertracking\fibers',num2str(j),'.tif');
%         print(h1,fullfile(figurepath),'-dtiff','-r100');
%         A=imread(figurepath);
%         imwrite(A,cat(2,paths{i},'\fibertracking\fibers.tif'),'WriteMode','append');
%         delete (figurepath);
%     end
end