%Author: Irene Wang
function movies = make_movies(path)
load('RedBlueColormaps.mat')
load('BlackColormaps.mat')
load(fullfile(path, 'Allresults2.mat'));

%% define video parameters
max_disp =2e-6;
max_traction = 2000;
length_video = size(Dx,3);
length_x = size(Dx,1);
length_y = size(Dx,2);
[PosX,PosY]=meshgrid(1:length_y,1:length_x);


h1=figure('units','Normalized','position',[0.45 0.05 0.4 0.4],'Name','Stress magnitude');
h2=figure('units','Normalized','position',[0.02 0.52 0.4 0.4],'Name','Displacements');
h3=figure('units','Normalized','position',[0.45 0.32 0.4 0.4],'Name','Force Orientation Map');

%% Displacement video
for i=1:length_video
displacement = sqrt(Dx(:,:,i).^2+Dy(:,:,i).^2);
figure(h2) % Displacements
imagesc(displacement),colormap(blackcmap),colorbar, caxis([0 max_disp]), axis off, daspect([1 1 1]), %.*BWgrid
hold on;

if i == 1
    quiver(PosX(1:4:end,1:4:end),PosY(1:4:end,1:4:end),10e6*Dx(1:4:end,1:4:end,i),10e6*Dy(1:4:end,1:4:end,i),'r','AutoScale','off');
else
    quiver(PosX(1:4:end,1:4:end),PosY(1:4:end,1:4:end),10e6*Dx(1:4:end,1:4:end,i),10e6*Dy(1:4:end,1:4:end,i),'r','AutoScale','off');
end

if ~exist(fullfile(path,'movies'),'dir')
    mkdir(fullfile(path,'movies'))
end

figurepath=cat(2,path,'\movies\displacement',num2str(i),'.tif');
print(h2,fullfile(figurepath),'-dtiff','-r100');
A=imread(figurepath);
imwrite(A,cat(2,path,'\movies\displacement.tif'),'WriteMode','append');
delete (figurepath)
end

%% Traction force video
for i=1:length_video
traction = sqrt(Tx(:,:,i).^2+Ty(:,:,i).^2);
figure(h1) % Displacements
imagesc(traction),colormap(jet),colorbar, caxis([0 max_traction]), axis off, daspect([1 1 1]), %.*BWgrid
hold on;

if i == 1
    quiver(PosX(1:4:end,1:4:end),PosY(1:4:end,1:4:end),1e-2*Tx(1:4:end,1:4:end,i),1e-2*Ty(1:4:end,1:4:end,i),'r','AutoScale','off');
else
    quiver(PosX(1:4:end,1:4:end),PosY(1:4:end,1:4:end),1e-2*Tx(1:4:end,1:4:end,i),1e-2*Ty(1:4:end,1:4:end,i),'r','AutoScale','off');
end
if ~exist(fullfile(path,'movies'),'dir')
    mkdir(fullfile(path,'movies'))
end

figurepath=cat(2,path,'\movies\traction',num2str(i),'.tif');
print(h1,fullfile(figurepath),'-dtiff','-r100');
A=imread(figurepath);
imwrite(A,cat(2,path,'\movies\traction.tif'),'WriteMode','append');
delete (figurepath)
end

% %% FOM video
% for i=1:length_video
% Angle2=atan(abs(Ty(:,:,i)./Tx(:,:,i)));
% traction = sqrt(Tx(:,:,i).^2+Ty(:,:,i).^2);
% 
% AmpT=traction/(max(traction(:))-500); % this adjusts the intensity. There is a saturation with a lot of values at 1 on purpose, since it is nicer to look at.
% ind=find(AmpT>1);
% AmpT(ind)=1;
% 
% angle_255=floor(Angle2/max(Angle2(:))*255)+1;         % Norm the angle to be between 1 and 256
% 
% DIMS=size(angle_255);
% cm=jet(256);
% 
% R=reshape(cm(angle_255,1),DIMS);                    % Convert to RGB colors
% G=reshape(cm(angle_255,2),DIMS);
% B=reshape(cm(angle_255,3),DIMS);
% 
% RGB=zeros(DIMS(1),DIMS(2),3);
% RGB(:,:,1)=R.*AmpT;
% RGB(:,:,2)=G.*AmpT;
%  RGB(:,:,3)=B.*AmpT;                                         
%            
% figure(h3);
% imshow(RGB,[],'InitialMagnification','fit')
% colormap(jet)
% colorbar;  caxis([0 90])% Display FOM
% c = colorbar;
% c.Label.String = 'Angle of traction force in degree';
% axis equal
% set(gca,'Visible','off');
% 
% 
% figurepath=cat(2,path,'\movies\FOM',num2str(i),'.tif');
% print(h3,fullfile(figurepath),'-dtiff','-r100');
% A=imread(figurepath);
% imwrite(A,cat(2,path,'\movies\FOM.tif'),'WriteMode','append');
% delete (figurepath)
% end
end