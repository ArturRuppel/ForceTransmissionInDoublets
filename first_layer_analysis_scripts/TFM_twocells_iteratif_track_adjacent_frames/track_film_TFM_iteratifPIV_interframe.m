%Author: Irène Wang
%Changes on 11/04/2016: - introduce overlap between tracked windows
%change on 26/10/2016: iterative PIV before tracking
%from Cécile version (stress variation removed)
%tracking between adjacent frames
%% Load films
function track = track_film_TFM_iteratifPIV_interframe(beads_file,beads_path,initial_file,initial_path, brightfield_file, brightfield_path, mask_left_path, mask_left_file, mask_right_path, mask_right_file)
close all


load('RedBlueColormaps.mat')
load('BlackColormaps.mat')

if ~exist('path','var')
path='D:';
end

if initial_file
    path=initial_path;
    nonstressed=imread(fullfile(initial_path,initial_file));

    if beads_file
        path = beads_path;
        filename=beads_file;
        info = imfinfo(fullfile(path,filename),'tif');
        Nb_frames=numel(info);
        %Nb_frames=10;
        I=imread(fullfile(path,filename),'tif');

        cl=class(I);
        stressed=zeros([size(I) Nb_frames], cl);
        stressed(:,:,1)=I;
        for frame=2:Nb_frames
            stressed(:,:,frame)=imread(fullfile(path,filename),frame);
        end
    end
else
    path = beads_path;
    filename=beads_file;
    info = imfinfo(fullfile(path,filename),'tif');
    Nb_frames=numel(info);
    %Nb_frames=10;
    I=imread(fullfile(path,filename),'tif',2);

    cl=class(I);
    stressed=zeros([size(I) Nb_frames-1], cl);
    stressed(:,:,1)=I;
    for frame=2:Nb_frames-1
        stressed(:,:,frame)=imread(fullfile(path,filename),frame+1);
    end
    nonstressed=imread(fullfile(path,filename),1);
end



if brightfield_file
    info = imfinfo(fullfile(brightfield_path,brightfield_file),'tif');
    Nb_frames=numel(info);
    I=imread(fullfile(brightfield_path,brightfield_file),'tif');
    cellule=zeros([size(I) Nb_frames], class(I));
    cellule(:,:,1)=contrast(I,0.1,0.1);
    for frame=2:Nb_frames
        cellule(:,:,frame)=contrast(imread(fullfile(brightfield_path,brightfield_file),frame),0.1,0.1);
    end
else
    cellule=stressed;
end

if mask_left_file
    I=imread(fullfile(mask_left_path,mask_left_file),'tif');
    centro_left=zeros([size(I) Nb_frames], class(I));
    centro_left(:,:,1)=I;
    no_images=numel(imfinfo(cat(2,mask_left_path,mask_left_file)));
    for frame=2:Nb_frames
        if no_images==1
            centro_left(:,:,frame)=imread(fullfile(mask_left_path,mask_left_file));
        else
            centro_left(:,:,frame)=imread(fullfile(mask_left_path,mask_left_file),frame);
        end
    end
else
    centro_left=ones([size(I) Nb_frames], class(I));    
end

if mask_right_file
    I=imread(fullfile(mask_right_path,mask_right_file),'tif');
    centro_right=zeros([size(I) Nb_frames], class(I));
    centro_right(:,:,1)=I;
    no_images=numel(imfinfo(cat(2,mask_right_path,mask_right_file)));
    for frame=2:Nb_frames
        if no_images==1
            centro_right(:,:,frame)=imread(fullfile(mask_right_path,mask_right_file));
        else
            centro_right(:,:,frame)=imread(fullfile(mask_right_path,mask_right_file),frame);
        end
    end
else
    centro_right=ones([size(I) Nb_frames], class(I));   
end


% choice=questdlg('Do you want to make movies at the end of analysis?','Movies','yes','no','no');
%     if strcmp(choice,'yes')
%         movies=true;
%     else
%         movies=false;
%     end

%% Begin analysis

%%%%%%%% Parameters to play with %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pix = 0.108e-6;% taille du pixel %60x 0.091 - 40x 0.133 
E=20000;
nu=0.5; %Poisson ratio
alphadef=0.2e-19;   %*(100000/E)^2; %this coef of regularization should vary with the young modulus _TOCHECK!!!
%initial window size
window=128;
%number of iterations
iter=2;%after each interation the window size is divided by 2
overlapPIV=64;%in pixels (applied to first window)                  %32
overlapTrack=32;%in pixels (applied to last window befor tracking)  %16
interval=8;
pas=interval*pix;
%Tracking parameters 
featsize=2;% rayon de la particule en pixel, detect?e par un masque de taille featsize typiquement 2 ou 3
barrg=3; % rayon de giration (related to the size of the particle)
barcc=0.25; % excentricit? (0 = rond 1 = pas rod du tout) typiquement entre 0.1 OU 0.2
IdivRg=0;%intensit? int?gr?e minimale pour detecter une particule
masscut=100000; %250000
maxd=2;%deplacement max d'une bille entre les 2 images
r_neighbor = 50; % in pixel
% corr_th = 0.5;
dev_th = 1.0e-6; % if the mean deviation of a bead vector with it's neigboring vector is higher then this, the bead is rejected
disp_th = 2.5e-6; % if the measured displacement of a bead is higher than this value, the bead is rejected

% %%%%%%%% Parameters to play with %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% pix = 0.302e-6;% taille du pixel %60x 0.091 - 40x 0.133 
% E=5000;
% nu=0.5; %Poisson ratio
% alphadef=1e-19;   %*(100000/E)^2; %this coef of regularization should vary with the young modulus _TOCHECK!!!
% %initial window size
% window=128;
% %number of iterations
% iter=1;%after each interation the window size is divided by 2
% overlapPIV=64;%in pixels (applied to first window)                  %32
% overlapTrack=32;%in pixels (applied to last window befor tracking)  %16
% interval=8;
% pas=interval*pix;
% %Tracking parameters 
% featsize=3;% rayon de la particule en pixel, detect?e par un masque de taille featsize typiquement 2 ou 3
% barrg=5; % rayon de giration (related to the size of the particle)
% barcc=0.2; % excentricit? (0 = rond 1 = pas rod du tout) typiquement entre 0.1 OU 0.2
% IdivRg=0;%intensit? int?gr?e minimale pour detecter une particule
% masscut=10000; %250000
% maxd=4;%deplacement max d'une bille entre les 2 images
% r_neighbor = 25; % in pixel
% corr_th = 0.5;

%Display parameters
% max_disp = 1e-6; %m
% max_strain = 0.2; %no unit
% max_stress = 2000; %Pa

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

h1=figure('units','Normalized','position',[0.02 0.05 0.9 0.83],'Name','Tracking');
h2=figure('units','Normalized','position',[0.02 0.05 0.4 0.4],'Name','Stress vectors');
h3=figure('units','Normalized','position',[0.45 0.05 0.4 0.4],'Name','Stress magnitude');
h4=figure('units','Normalized','position',[0.02 0.52 0.4 0.4],'Name','Displacements');
h5=figure('units','Normalized','position',[0.45 0.32 0.4 0.4],'Name','Force Orientation Map');

%decalages par rapport à l'image nonstressed
Regx=zeros(1,Nb_frames);
Regy=zeros(1,Nb_frames);
%partie subpixel du decalage
Restex=zeros(1,Nb_frames);
Restey=zeros(1,Nb_frames);

%Calculated parameters
U=zeros(1,Nb_frames);%contractile energy
Pmax=zeros(1,Nb_frames);%maximum stress
Ftot=zeros(1,Nb_frames);% Sum of forces in absolute values
Pmoy=zeros(1,Nb_frames);%average stress
Dmoy=zeros(1,Nb_frames);%average displacement
Fvect=zeros(1,Nb_frames);% sum of forces in vectors
DispMax=zeros(1,Nb_frames);%maximum displacement (after smoothing) 
StrainMax=zeros(1,Nb_frames);%maximum displacement (after smoothing)
Moment=zeros(1,Nb_frames);%contactile moment
Angle=zeros(1,Nb_frames);%angle of max contraction crelative to vertical
Polar=zeros(1,Nb_frames);%degree of polarization
Fcellcell=zeros(1,Nb_frames);%intercellular force
DFcellcell=zeros(1,Nb_frames);%intercellular force difference between right and left cell
Totalcell1=zeros(1,Nb_frames);% Somme en norme des tractions cellule1-substrat
Totalcell2=zeros(1,Nb_frames);% Somme en norme des tractions cellule2-substrat
Cell_ECMratio1=zeros(1,Nb_frames);%cell-cell force to cell-ECM force ratio, first cell
Cell_ECMratio2=zeros(1,Nb_frames);%cell-cell force to cell-ECM force ratio, second cell
Forcex_cell1=zeros(1,Nb_frames);%sum of x component of forces (in magnitude) for cell 1
Forcey_cell1=zeros(1,Nb_frames);%sum of y component of forces (in magnitude) for cell 1
Forcex_cell2=zeros(1,Nb_frames);%sum of x component of forces (in magnitude) for cell 2
Forcey_cell2=zeros(1,Nb_frames);%sum of y component of forces (in magnitude) for cell 2
Deplx_cell1=zeros(1,Nb_frames);%sum of x component of displacements (in magnitude) for cell 1
Deply_cell1=zeros(1,Nb_frames);%sum of y component of displacements (in magnitude) for cell 1
Deplx_cell2=zeros(1,Nb_frames);%sum of x component of displacements (in magnitude) for cell 2
Deply_cell2=zeros(1,Nb_frames);%sum of y component of displacements (in magnitude) for cell 2

%% Registration of all images
nim=cell(1,Nb_frames);
sim=cell(1,Nb_frames);
disp('Registration of all images (may take some time)...')
for ff=1:Nb_frames  
    
if ff==1
    [nim{1},sim{1},regx,regy]=registration_ssinterp_beads(nonstressed,stressed(:,:,1));
    Regx(1)=regx;
    Regy(1)=regy;
    Restex(1)=regx-round(regx);
    Restey(1)=regy-round(regy);
else
    [nim{ff},sim{ff},regx,regy]=registration_ssinterp_beads(stressed(:,:,ff-1),stressed(:,:,ff));
    Regx(ff)=Regx(ff-1)+regx;
    Regy(ff)=Regy(ff-1)+regy;
    Restex(ff)=regx-round(regx);
    Restey(ff)=regy-round(regy);
end
end

if ~isempty(Regx(Regx>0)), maxdxp=max(round(Regx(Regx>0))); else maxdxp=0; end
if ~isempty(Regx(Regx<0)), maxdxn=min(round(Regx(Regx<0))); else maxdxn=0; end
if ~isempty(Regx(Regy>0)), maxdyp=max(round(Regy(Regy>0))); else maxdyp=0; end
if ~isempty(Regx(Regy<0)), maxdyn=min(round(Regy(Regy<0))); else maxdyn=0; end

n=nonstressed(maxdyp+1:size(nonstressed,1)-abs(maxdyn),maxdxp+1:size(nonstressed,2)-abs(maxdxn));
Mask_left=false([size(n) Nb_frames]);
Mask_right=false([size(n) Nb_frames]);
Brightfield=uint8(zeros([size(n) Nb_frames]));

% Mask and brightfield images must be cut to the same size according to shifts
for ff=1:Nb_frames
    
    Mask_left(:,:,ff)=logical(centro_left(maxdyp-round(Regy(ff))+1:size(nonstressed,1)-(abs(maxdyn)+round(Regy(ff))),maxdxp-round(Regx(ff))+1:size(nonstressed,2)-(abs(maxdxn)+round(Regx(ff)))));
    Mask_right(:,:,ff)=logical(centro_right(maxdyp-round(Regy(ff))+1:size(nonstressed,1)-(abs(maxdyn)+round(Regy(ff))),maxdxp-round(Regx(ff))+1:size(nonstressed,2)-(abs(maxdxn)+round(Regx(ff)))));
    bf=cellule(maxdyp-round(Regy(ff))+1:size(nonstressed,1)-(abs(maxdyn)+round(Regy(ff))),maxdxp-round(Regx(ff))+1:size(nonstressed,2)-(abs(maxdxn)+round(Regx(ff))),ff);
    Brightfield(:,:,ff)=uint8(round(double(bf-min(bf(:)))/double(max(bf(:))-min(bf(:)))*255));
end
disp('Done!')
[dimy,dimx]=size(n);
period=window-overlapPIV;
nbx=floor((dimx-window)/period)+1;
nby=floor((dimy-window)/period)+1;
[posx,posy]=meshgrid((0:nbx-1)*period+floor((dimx-(nbx-1)*period)/2)+1,(0:nby-1)*period+floor((dimy-(nby-1)*period)/2)+1);
nodeX=floor(period*nbx/interval);
nodeY=floor(period*nby/interval);
[gridX,gridY]=meshgrid((0:nodeX-1)*interval+floor((dimx-nbx*period)/2)+1+interval/2,(0:nodeY-1)*interval+floor((dimy-nby*period)/2)+1+interval/2);

%%
Dx=zeros([size(gridX) Nb_frames]);
Dy=zeros([size(gridX) Nb_frames]);
Tx=zeros([size(gridX) Nb_frames]);
Ty=zeros([size(gridX) Nb_frames]);

for ff=1:Nb_frames 
if ff>1
iter=0;%after each interation the window size is divided by 2
maxd=1;
end    
    %add condition if the result file .mat exists, load it is skip the
    %computation of the displacements matrices (deplacement*)
    
% if exist(fullfile(path,[filename(1:end-4),'_results', num2str(ff),'.mat']),'file') > 0, 
%     load(fullfile(path,[filename(1:end-4),'_results', num2str(ff),'.mat']));  
%         % use the existing displacement already computed    
%         % !!! ONLY if you don't change the tracking parameters !!! 
% else 
        
    disp(['Tracking: image ',num2str(ff)]);
    if stressed(:,:,ff)==0
        disp('Null image!!!')
       return
    end
    restex=Restex(ff);
    restey=Restey(ff);
    mask_corr_left=Mask_left(:,:,ff);
    mask_corr_right=Mask_right(:,:,ff);
    bf=Brightfield(:,:,ff);
    
    
mask_corr=mask_corr_left|mask_corr_right;
    
%gridframe are the grid positions in the reference of this frame
if ff==1
    if round(Regx(1))<0
        posframex=posx+maxdxp;
        gridframeX=gridX+maxdxp;
    else
        posframex=posx+maxdxp-round(Regx(1));
        gridframeX=gridX+maxdxp-round(Regx(1));
    end
    if round(Regy(1))<0
        posframey=posy+maxdyp;
        gridframeY=gridY+maxdyp;
    else
        posframey=posy+maxdyp-round(Regy(1));
        gridframeY=gridY+maxdyp-round(Regy(1));
    end
else
    %shift already corrected on the images
    decx=Regx(ff)-Regx(ff-1);
    decy=Regy(ff)-Regy(ff-1);
    if decx<0
        posframex=posx+maxdxp-round(Regx(ff)+decx);
        gridframeX=gridX+maxdxp-round(Regx(ff)+decx);
    else 
        posframex=posx+maxdxp-round(Regx(ff));
        gridframeX=gridX+maxdxp-round(Regx(ff));
    end
    if decy<0
        posframey=posy+maxdyp-round(Regy(ff)+decy);
        gridframeY=gridY+maxdyp-round(Regy(ff)+decy);
    else 
        posframey=posy+maxdyp-round(Regy(ff));
        gridframeY=gridY+maxdyp-round(Regy(ff));
    end
end

    n=nim{ff};
    s=sim{ff};
deplaceX=zeros(10000,1);
deplaceY=zeros(10000,1);
positionX=zeros(10000,1);
positionY=zeros(10000,1);
compteur=0;
mauvaises_fenetres=0;
[M,N]=size(s);

disp(['Nombre de fenetres: ',num2str(nby*nbx)]);
for i=1:nbx  
for j=1:nby
         mauvais_track=false;
       
       %nombre de fenêtres
       Nwin=2.^(2*(0:iter));  
       %tailles des fenêtres
       mi=window./(2.^(0:iter));
       %tableau de positions (centre de fenêtres): à chaque étape le nb de positions est
       %multiplié par 4
       posix=cell(1,iter+1);
       posiy=cell(1,iter+1);
       posix{1}=posframex(1,i);
       posiy{1}=posframey(j,1);
       % tableau d'offsets
       offsetx=cell(1,iter+1);
       offsety=cell(1,iter+1);
       offsetx{1}=0;
       offsety{1}=0;
     
       
       for k=0:iter
           %définit les offsets et les positions pour chaque itération
           offsetx{k+1}=zeros(2^k);
           offsety{k+1}=zeros(2^k);
           %recopie les valeurs d'avant
           for l=1:2^(k-1)
               for m=1:2^(k-1)
               offsetx{k+1}((l-1)*2+1:l*2,(m-1)*2+1:m*2)=offsetx{k}(l,m)*ones(2);
               offsety{k+1}((l-1)*2+1:l*2,(m-1)*2+1:m*2)=offsety{k}(l,m)*ones(2);

               end
           end
           [px,py]=meshgrid((-mi(k+1)/2*(2^k-1):mi(k+1):mi(k+1)/2*(2^k-1)),(-mi(k+1)/2*(2^k-1):mi(k+1):mi(k+1)/2*(2^k-1)));
           posix{k+1}=posframex(1,i)+px;
           posiy{k+1}=posframey(j,1)+py;

           for nw=1:Nwin(k+1) 
              
               
               [sub1,sub2]=ind2sub([2^k 2^k],nw);
               %1. interpolation de n
               if k>0
            xi=(1:mi(k+1))-offsetx{k+1}(sub1,sub2)+posix{k+1}(sub1,sub2)-mi(k+1)/2-1;
            yi=(1:mi(k+1))-offsety{k+1}(sub1,sub2)+posiy{k+1}(sub1,sub2)-mi(k+1)/2-1;
            n_win = interp2(double(n),xi',yi);
               else
            n_win=n(posiy{k+1}(sub1,sub2)-mi(k+1)/2:posiy{k+1}(sub1,sub2)+mi(k+1)/2-1,posix{k+1}(sub1,sub2)-mi(k+1)/2:posix{k+1}(sub1,sub2)+mi(k+1)/2-1);
               end
           %2. mesure du décalage
          
            s_win=s(posiy{k+1}(sub1,sub2)-mi(k+1)/2:posiy{k+1}(sub1,sub2)+mi(k+1)/2-1,posix{k+1}(sub1,sub2)-mi(k+1)/2:posix{k+1}(sub1,sub2)+mi(k+1)/2-1); 
%            n_win=n(posy(j,1)-window/2:posy(j,1)+window/2-1,posx(1,i)-window/2:posx(1,i)+window/2-1);
%            s_win=s(posy(j,1)-window/2:posy(j,1)+window/2-1,posx(1,i)-window/2:posx(1,i)+window/2-1); 
        [offx,offy,Cmax]=decalage(s_win,n_win);
         deltaI=min((max(n_win(:))-min(n_win(:)))/mean(n_win(:)),(max(s_win(:))-min(s_win(:)))/mean(s_win(:)));
         if deltaI<5||Cmax<0.5%||sqrt(offx^2+offy^2)>mi(k+1)/3
             offx=0;
             offy=0;
         end
%          figure
%             subplot(1,2,1), imshow(s_win,[]), 
%             subplot(1,2,2), imshow(n_win,[]),title(['k=',num2str(k),'- (',num2str(sub1),',',num2str(sub2),')'])
%             hold on
%             quiver(mi(k+1)/2,mi(k+1)/2,offx,offy)
%             hold off

         %remplissage du tableau de valeurs
         if k==0
         offsetx{1}=offx;
         offsety{1}=offy; 
         else
            offsetx{k+1}(sub1,sub2)=offx+ offsetx{k}(ceil(sub1/2),ceil(sub2/2));
            offsety{k+1}(sub1,sub2)=offy+ offsety{k}(ceil(sub1/2),ceil(sub2/2));
%          [sub1,sub2]=ind2sub([2^k 2^k],nw);
%          offsetx(1+(sub1-1)*2^(iter-k):(sub1)*2^(iter-k),1+(sub2-1)*2^(iter-k):(sub2)*2^(iter-k))=offsetx(1+(sub1-1)*2^(iter-k):(sub1)*2^(iter-k),1+(sub2-1)*2^(iter-k):(sub2)*2^(iter-k))+offx;
%          offsety(1+(sub1-1)*2^(iter-k):(sub1)*2^(iter-k),1+(sub2-1)*2^(iter-k):(sub2)*2^(iter-k))=offsety(1+(sub1-1)*2^(iter-k):(sub1)*2^(iter-k),1+(sub2-1)*2^(iter-k):(sub2)*2^(iter-k))+offy;
         end
           
         %après la dernière itération, faire le tracking pour chaque
         %fenetre élargie
         if k==iter
             
            
         % interpolation de n
         win=mi(k+1)+overlapTrack;
         ystart=posiy{k+1}(sub1,sub2)-win/2;
         xstart=posix{k+1}(sub1,sub2)-win/2;
         yend=posiy{k+1}(sub1,sub2)+win/2-1;
         xend=posix{k+1}(sub1,sub2)+win/2-1;
         if ystart<1, ystart=1;end
         if xstart<1, xstart=1;end
         if yend>M, yend=M;end
         if xend>N, xend=N;end
            xi=(1:win)-offsetx{k+1}(sub1,sub2)+xstart-1;
            yi=(1:win)-offsety{k+1}(sub1,sub2)+ystart-1;
            n_interp = interp2(double(n),xi',yi);
           s_win=s(ystart:yend,xstart:xend); 
            
         %tracking
              mauvais_track=0; 
       if strcmp(cl,'uint16')
       %n_win=uint16(n_interp);
       n_win= uint16(65535/(max(n_interp(:))-min(n_interp(:)))*(n_interp-min(n_interp(:))));
       s_win= uint16(65535/double(max(s_win(:))-min(s_win(:)))*double(s_win-min(s_win(:))));
       elseif strcmp(cl,'uint8')
       n_win= uint8(255/(max(n_interp(:))-min(n_interp(:)))*(n_interp-min(n_interp(:))));
       s_win= uint8(255/double(max(s_win(:))-min(s_win(:)))*double(s_win-min(s_win(:))));%n_win=uint8(n_interp);
       end
       %Tracking sur les petites fen?tres
        nM = feature2D(n_win,1,featsize,masscut);
 %figure(5),imshow(n_win,[]), hold on, plot(nM(:,1),nM(:,2),'go'), hold off           
        if numel(nM)==1||isempty(nM)
            figure(16)
            imshow(n_win,[])
            mauvais_track=true;
        else
        %Rejection process
        X= nM(:,5)>barcc;
        nM(X,1:5)=0;
        X= nM(:,4)>barrg;
        nM(X,1:5)=0;
        X= nM(:,3)./nM(:,4)<IdivRg;
        nM(X,1:5)=0;
        nM=nM(nM(:,1)~=0,:);
        if numel(nM)==1||isempty(nM)
            figure(16)
            imshow(n_win,[])
            mauvais_track=true;
        end
        end
 %figure,imshow(n_win,[]), hold on, plot(nM(:,1),nM(:,2),'go'), hold off
        sM = feature2D(s_win,1,featsize,masscut);     
 %figure,imshow(s_win,[]), hold on, plot(sM(:,1),sM(:,2),'go'), hold off
 %                    waitforbuttonpress
        if numel(sM)==1||isempty(sM)
            figure(16)
            imshow(s_win,[])
            mauvais_track=true;
        else
        %Rejection process
        X= sM(:,5)>barcc;
        sM(X,1:5)=0;
        X= sM(:,4)>barrg;
        sM(X,1:5)=0;
        X= sM(:,3)./sM(:,4)<IdivRg;
        sM(X,1:5)=0;
        sM=sM(sM(:,1)~=0,:);
        if numel(sM)==1||isempty(sM)
            figure(16)
            imshow(s_win,[])
            mauvais_track=true;
        end
        end
%         figure
%             subplot(1,2,1), imshow(s_win,[]), hold on, plot(sM(:,1),sM(:,2),'go'), hold off
%             subplot(1,2,2), imshow(n_win,[]), hold on, plot(nM(:,1),nM(:,2),'go'), hold off
        if not(mauvais_track)
        Mtrack=[nM(:,1:2),ones(size(nM,1),1),ones(size(nM,1),1); sM(:,1:2),2*ones(size(sM,1),1),2*ones(size(sM,1),1)];
        [lub] = trackmem(Mtrack,maxd,2,2,0);        
        if lub==-1
            figure(15)
            subplot(1,2,1), imshow(s_win,[]), hold on, plot(sM(:,1),sM(:,2),'go'), hold off
            subplot(1,2,2), imshow(n_win,[]), hold on, plot(nM(:,1),nM(:,2),'go'), hold off
            mauvais_track=true;
        end
        end
        %numero des images: null=1, stressed=2
        if mauvais_track
        mauvaises_fenetres=mauvaises_fenetres+1;
%          deplaceX(compteur+1:compteur+2)=offsetx+restex;
%          deplaceY(compteur+1:compteur+2)=offsety+restey;
%          positionX(compteur+1:compteur+2)=posx(1,i);
%          positionY(compteur+1:compteur+2)=posy(j,1);
%          compteur=compteur+1;
        else
        nb_part=max(lub(:,5));
      
        deplaceX(compteur+1:compteur+nb_part)=lub((lub(:,4)==2),1)-lub((lub(:,4)==1),1)+offsetx{k+1}(sub1,sub2)+Restex(ff);
         deplaceY(compteur+1:compteur+nb_part)=lub((lub(:,4)==2),2)-lub((lub(:,4)==1),2)+offsety{k+1}(sub1,sub2)+Restey(ff);
         positionX(compteur+1:compteur+nb_part)=lub((lub(:,4)==2),1)+(posix{k+1}(sub1,sub2)-win/2)-1;
         positionY(compteur+1:compteur+nb_part)=lub((lub(:,4)==2),2)+(posiy{k+1}(sub1,sub2)-win/2)-1;
        compteur=compteur+nb_part;
        end
         end
           end
       end
end   
end
disp([num2str(compteur-mauvaises_fenetres) ' features tracked']);
disp([num2str(mauvaises_fenetres) ' mauvaises fenetres']);
deplaceX(compteur+1:10000)=[];
deplaceY(compteur+1:10000)=[];
positionX(compteur+1:10000)=[];
positionY(compteur+1:10000)=[];
%remove duplicates
Mpos=[positionX positionY];
[A,ind]=sortrows(Mpos);
deplaceX=deplaceX(ind);
deplaceY=deplaceY(ind);
tolerance=2;
diff=A(1:end-1,:)-A(2:end,:);
duplicate=find((diff(:,1).^2+diff(:,2).^2)<tolerance);
A(duplicate,:)=[];
positionX=A(:,1);
positionY=A(:,2);
deplaceX(duplicate)=[];
deplaceY(duplicate)=[];
disp(['Particules trackees (non duplicates): ', num2str(length(deplaceX))])

if ff==1
    filtergraph = figure;
    nb_beads = size(positionX);
    deplaceXh = deplaceX;
    deplaceYh = deplaceY;
    quiver(positionX,positionY,3*deplaceX,3*deplaceY,'r','AutoScale','off');hold on;
    for i = 1:nb_beads        
        index_neighbors = find(sqrt((positionX-positionX(i)).^2+(positionY-positionY(i)).^2)<r_neighbor);
        allbutone = find(index_neighbors~=i);
        index_neighbors = index_neighbors(allbutone);
        % allscalarp = deplaceXh(i)*deplaceXh(index_neighbors)+deplaceYh(i)*deplaceYh(index_neighbors); %calculate scalar product of the vector from each of it's neighbors
        % r_corr = mean(allscalarp./(sqrt(deplaceXh(i)^2+deplaceYh(i)^2)*sqrt(deplaceXh(index_neighbors).^2+deplaceYh(index_neighbors).^2))); %divide by length to obtain angle in between vectors, i.e. correlation coefficient
        
        alldev = sqrt((deplaceXh(i)-deplaceXh(index_neighbors)).^2+(deplaceYh(i)-deplaceYh(index_neighbors)).^2);
        meandev = mean(alldev);
        if meandev > dev_th/pix
            deplaceX(i) = mean(deplaceX(index_neighbors));
            deplaceY(i) = mean(deplaceY(index_neighbors));            
            plot(positionX(i),positionY(i),'*','Color','green');hold on;
        end

        if sqrt(deplaceX(i)^2+deplaceY(i)^2)>disp_th/pix
            deplaceX(i) = mean(deplaceX(index_neighbors));
            deplaceY(i) = mean(deplaceY(index_neighbors));
            plot(positionX(i),positionY(i),'*','Color','blue');hold on;
        end
%         if  r_corr < corr_th            
%             plot(positionX(i),positionY(i),'*','Color','blue');hold on;
%         end
    end
    print(filtergraph,cat(2,path,'filters'),'-dpng');
end
% if ff==1
%     filtergraph = figure;
%     nb_beads = size(positionX);
%     deplaceXh = deplaceX;
%     deplaceYh = deplaceY;
%     quiver(positionX,positionY,3*deplaceX,3*deplaceY,'r','AutoScale','off');hold on;
%     for i = 1:nb_beads        
%         index_neighbors = find(sqrt((positionX-positionX(i)).^2+(positionY-positionY(i)).^2)<r_neighbor);
%         allbutone = find(index_neighbors~=i);
%         index_neighbors = index_neighbors(allbutone);
%         r_corr = mean((deplaceXh(i)*deplaceXh(index_neighbors)+deplaceYh(i)*deplaceYh(index_neighbors)));%./(sqrt(deplaceXh(i)^2+deplaceYh(i)^2)*sqrt(deplaceXh(index_neighbors).^2+deplaceYh(index_neighbors).^2)));
%         if r_corr < corr_th
%             deplaceX(i) = mean(deplaceX(index_neighbors));
%             deplaceY(i) = mean(deplaceY(index_neighbors));            
%             plot(positionX(i),positionY(i),'*','Color','green');hold on;
%         end
% %         if sqrt(deplaceX(i)^2+deplaceY(i)^2)>maxd_filter/pix
% %             deplaceX(i) = mean(deplaceX(index_neighbors));
% %             deplaceY(i) = mean(deplaceY(index_neighbors));
% %             plot(positionX(i),positionY(i),'*','Color','blue');hold on;
% %         end
%     end
%     print(filtergraph,cat(2,path,'filters'),'-dpng');
% end


figure(h1) % Tracking
Idisp=zeros([size(s) 3]);
sn=double(s)/double(max(s(:)));
nn=double(n)/double(max(n(:)));
Idisp(:,:,1)=nn;
Idisp(:,:,2)=sn;
Idisp(:,:,3)=nn;
imshow(Idisp,[],'InitialMagnification','fit'),colormap(gray)
title(['Frame: ',num2str(ff),' / ', num2str(Nb_frames)]);
hold on
% if ff==1
    quiver(positionX,positionY,3*deplaceX,3*deplaceY,'r','AutoScale','off');
% else
%     quiver(positionX,positionY,10*deplaceX,10*deplaceY,'r','AutoScale','off');
% end
%hold on
% for i=6
%     for j=2
%         hold on
% plot([posx(1,i)-window/2 posx(1,i)-window/2 posx(1,i)+window/2 posx(1,i)+window/2 posx(1,i)-window/2],[posy(j,1)-window/2 posy(j,1)+window/2 posy(j,1)+window/2 posy(j,1)-window/2 posy(j,1)-window/2],'g')
%     end
% end
hold off
drawnow

if ~exist(fullfile(path,'figureTFM'),'dir')
    mkdir(fullfile(path,'figureTFM'))
end

figurepath=cat(2,path,'\figureTFM\Tracking',num2str(ff),'.tif');
print(h1,fullfile(figurepath),'-dtiff','-r100');
A=imread(figurepath);
imwrite(A,cat(2,path,'\figureTFM\Tracking.tif'),'WriteMode','append');
delete (figurepath)
%print(h1,fullfile(path,['cell',num2str(ff),'_tracking.tif']),'-dtiff','-r600'); 
%print(h1,fullfile(path,'beads',['image',num2str(ff),'.tif']),'-dtiff','-r300');

% Mask
BW=logical(mask_corr);
% SE=strel('disk',10);
% BW2=imdilate(BW,SE);
BWgrid=logical(BW(gridY(:,1),gridX(1,:)));


%% Displacements

% nodeX=floor(window*nbx/interval);
% nodeY=floor(window*nby/interval);
% [gridX,gridY]=meshgrid((0:nodeX-1)*interval+floor((dimx-nbx*window)/2)+1+interval/2,(0:nodeY-1)*interval+floor((dimy-nby*window)/2)+1+interval/2);
if ff==1
deplacementXa=pix*griddata(positionX,positionY,deplaceX,gridframeX,gridframeY);
deplacementYa=pix*griddata(positionX,positionY,deplaceY,gridframeX,gridframeY);
deplacementXa(isnan(deplacementXa))=0;
deplacementYa(isnan(deplacementYa))=0;
else
%relative displacement from the previous frame    
ddX=pix*griddata(positionX,positionY,deplaceX,gridframeX,gridframeY);
ddY=pix*griddata(positionX,positionY,deplaceY,gridframeX,gridframeY);
ddX(isnan(ddX))=0;
ddY(isnan(ddY))=0;
deplacementXa=deplacementXa+ddX;
deplacementYa=deplacementYa+ddY;
end
%smoothing
wsmoothing=2;
[x,y]=meshgrid((-4*wsmoothing:4*wsmoothing),(-4*wsmoothing:4*wsmoothing));
Gauss=exp(-2*(x.^2+y.^2)/wsmoothing^2);
h=Gauss/sum(Gauss(:));
deplacementX=imfilter(deplacementXa,h);
deplacementY=imfilter(deplacementYa,h);

deplacement = sqrt(deplacementX.^2+deplacementY.^2);
deplacementCrop = deplacement.*BWgrid;

DispMax(ff) = max(max(deplacement.*BWgrid)); 
% end
%% Strain / Dilation
[dimx_strain, dimy_strain]=size(deplacement);


%Find the derivatives with respect to each direction using central
%difference method to get the deformation gradient tensor 

xY = zeros(dimx_strain, dimy_strain);
yY = zeros(dimx_strain, dimy_strain);
xX = zeros(dimx_strain, dimy_strain);
yX = zeros(dimx_strain, dimy_strain);

for i=2:dimx_strain-1,
    for j=2:dimy_strain-1,

        yX(i,j) = (deplacementX(i+1,j)-deplacementX(i-1,j))/(2*interval*pix);  %d/dY in direction x
        yY(i,j) = (deplacementY(i+1,j)-deplacementY(i-1,j))/(2*interval*pix)+1;  %d/dY in direction y
        
        xX(i,j) = (deplacementX(i,j+1)-deplacementX(i,j-1))/(2*interval*pix)+1;  %d/dX in direction x
        xY(i,j) = (deplacementY(i,j+1)-deplacementY(i,j-1))/(2*interval*pix);  %d/dX in direction y         
        
    end
end


% Use the derivatives to find the major and minor strain eigenvalues (lp and lm)
% and the x and y coordinates of the major eigenvector (eigenX and eigenY)

%The Lagrangian strain tensor is defined as E = (C - I)/2, where C = F'F is the right Cauchy-Green deformation tensor. 

E11=(xX.*xX+yX.*yX-1)/2;    %a
E22=(xY.*xY+yY.*yY-1)/2;
E12=(xX.*xY+yX.*yY)/2;      %b

Surf_Variation = E11 + E22; % >0 means expansion, <0 means compression 

%steps to diagonalize the matrix but useless to get the "dilatation" value
%(surface variation) 
I1=E11+E22;
I2=(E11.*E22)-(E12.*E12); 

eigenvalue_p =(I1+sqrt(I1.*I1-4*I2))/2;           
eigenvectorX_p = eigenvalue_p.*E12./sqrt(E12.*E12+(eigenvalue_p-E11).*(eigenvalue_p-E11));
eigenvectorY_p = eigenvalue_p.*(eigenvalue_p-E11)./sqrt(E12.*E12+(eigenvalue_p-E11).*(eigenvalue_p-E11));
                
eigenvalue_m =(I1-sqrt(I1.*I1-4*I2))/2;
eigenvectorX_m = eigenvalue_m.*E12./sqrt(E12.*E12+(eigenvalue_m-E11).*(eigenvalue_m-E11));
eigenvectorY_m = eigenvalue_m.*(eigenvalue_m-E11)./sqrt(E12.*E12+(eigenvalue_m-E11).*(eigenvalue_m-E11));


StrainMax(ff) = max(max(Surf_Variation.*BWgrid)); 
% end
%% Traction / Pressure

[Tractionx,Tractiony,mu,theta]=regularized(deplacementX,deplacementY,E,nu,pas,alphadef);




figure(h4) % Displacements
imagesc(deplacement),colormap(blackcmap),colorbar, caxis([0 2e-6]), axis off, daspect([1 1 1]), %.*BWgrid

%figurepath=cat(2,path,'\figureTFM\Displacement',num2str(ff),'.tif');
%print(h4,fullfile(figurepath),'-dtiff','-r100');
% A=imread(figurepath);
% imwrite(A,cat(2,path,'\figureTFM\Displacement.tif'),'WriteMode','append');
% delete (figurepath)
%save(fullfile(path,['cell',num2str(ff),'_Displacement.dat']),'deplacement','-ascii','-tabs')

% figure(h6) % Disp profile
% plot(gridX(end/2,end:-1:1)*pix,deplacement(end/2,end:-1:1)), %.*BWgrid(end/2,end:-1:1)
%hold on
%axis equal
%print(h4,fullfile(path,['cell',num2str((ff)),'_dispMap.tif']),'-dtiff','-r600');

% figure(h5) % Strains
% imagesc((Surf_Variation)),colormap(redbluecmap),colorbar, caxis([-max_strain max_strain]), axis off, daspect([1 1 1]), %.*BWgrid 

% Angle2=atan(abs(Tractiony./Tractionx));
% TMagn=sqrt(Tractionx.^2+Tractiony.^2);
% 
% AmpT=TMagn/(max(TMagn(:))-50); % this adjusts the intensity. There is a saturation with a lot of values at 1 on purpose, since it is nicer to look at.
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
           
% figure(h5);
% imshow(RGB,[],'InitialMagnification','fit')
% colormap(jet)
% colorbar;  caxis([0 90])% Display FOM
% c = colorbar;
% c.Label.String = 'Angle of traction force in degree';
% axis equal
% set(gca,'Visible','off');
% 
% 
% figurepath=cat(2,path,'\figureTFM\FOM',num2str(ff),'.tif');
% print(h5,fullfile(figurepath),'-dtiff','-r100');
% A=imread(figurepath);
% imwrite(A,cat(2,path,'\figureTFM\FOM.tif'),'WriteMode','append');
% delete (figurepath)
%save(fullfile(path,['cell',num2str(ff),'_Strain.dat']),'Surf_Variation','-ascii','-tabs')

%imagesc((xX+yY)),colormap(redbluecmap),colorbar, caxis([-max_strain max_strain]), axis off, daspect([1 1 1]), %.*BWgrid 


%% Calculations

Txcrop=Tractionx.*BWgrid;
Tycrop=Tractiony.*BWgrid;
    
%calculate all normal quantities (5 quantities)
%Contractile energy
pas=interval*pix;
u=Txcrop.*deplacementX+Tycrop.*deplacementY;  %produit scalaire traction par deplacement
U(ff)=pas^2/2*sum(u(:));

%Maximum stress
Tmagncrop=sqrt(Txcrop.^2+Tycrop.^2);
ntot=bwarea(BWgrid);
thres=max(Tmagncrop(:));
while numel(find(Tmagncrop>=thres))<0.05*ntot
    thres=thres-20;
end
Pmax(ff)=mean(Tmagncrop(find(Tmagncrop>=thres)));


%Total forces in magnitude
Ttot=sum(Tmagncrop(:));
Ftot(ff)=Ttot*(pix*interval)^2;
%disp(['Total des forces: ',  num2str(Ftot(ff)),' N']);

%Average stress 
Sgrid=pas^2;
Pmoy(ff)=Ttot/bwarea(BWgrid);% 

Dmoy(ff)=mean(mean(deplacementCrop));
%disp(['Contrainte moyenne: ',  num2str(TmagnMOY),' Pa']);

%Vectorial sum of forces
Fvect(ff)=sqrt((sum(Txcrop(:)))^2+(sum(Tycrop(:)))^2)*(pix*interval)^2;

%% Calculs pour chaque cellule du doublet et force cell-cell
disp('---- Calculs sur les cellules individuelles -----')
BW=mask_corr_left;
BWgrid1=logical(BW(gridY(:,1),gridX(1,:)));
BW=mask_corr_right;
BWgrid2=logical(BW(gridY(:,1),gridX(1,:)));
Txcrop1=Tractionx.*BWgrid1;
Tycrop1=Tractiony.*BWgrid1;
Txcrop2=Tractionx.*BWgrid2;
Tycrop2=Tractiony.*BWgrid2;

Tnorm=sqrt(Txcrop1.^2+Tycrop1.^2);
Totalcell1(ff)=sum(Tnorm(:))*Sgrid;% somme des forces de traction en norme
Tnorm=sqrt(Txcrop2.^2+Tycrop2.^2);
Totalcell2(ff)=sum(Tnorm(:))*Sgrid;% somme des forces de traction en norme

disp('---- Force cellule-cellule -----')
Fcellcell(ff)=sqrt((sum(Txcrop1(:))-sum(Txcrop2(:)))^2+(sum(Tycrop1(:))-sum(Tycrop2(:)))^2)*Sgrid/2;
disp(['Force cellule-cellule : ',  num2str(Fcellcell(ff)),' N']);
DFcellcell(ff)=sqrt((sum(Txcrop1(:))+sum(Txcrop2(:)))^2+(sum(Tycrop1(:))+sum(Tycrop2(:)))^2)*Sgrid/2;
disp(['Incertitude : ',  num2str(DFcellcell(ff)),' N']);
Cell_ECMratio1(ff)=sqrt((sum(Txcrop1(:)))^2+(sum(Tycrop1(:)))^2)*Sgrid/Totalcell1(ff);
Cell_ECMratio2(ff)=sqrt((sum(Txcrop2(:)))^2+(sum(Tycrop2(:)))^2)*Sgrid/Totalcell2(ff);
disp(['Rapport F cell-cell/F cell-ECM pour cellule 1: ',  num2str(Cell_ECMratio1(ff)*100),' %']);
disp(['Rapport F cell-cell/F cell-ECM pour cellule 2: ',  num2str(Cell_ECMratio2(ff)*100),' %']);
% %Calcul des moments
% s=regionprops(mask1_corr,'Area','Centroid');
% [~,ind]=max([s.Area]);
%  x1=s(ind).Centroid(:,1);
%  y1=s(ind).Centroid(:,2);
% s=regionprops(mask2_corr,'Area','Centroid');
% [~,ind]=max([s.Area]);
%  x2=s(ind).Centroid(:,1);
%  y2=s(ind).Centroid(:,2);
%  Mxx1=sum(sum((gridX-x1).*Txcrop1))*pix*Sgrid;
% Mxx2=sum(sum((gridX-x2).*Txcrop2))*pix*Sgrid;
% Myy1=sum(sum((gridY-y1).*Tycrop1))*pix*Sgrid;
% Myy2=sum(sum((gridY-y2).*Tycrop2))*pix*Sgrid;
% Mxy1=sum(sum((gridX-x1).*Tycrop1))*pix*Sgrid;
% Mxy2=sum(sum((gridX-x1).*Tycrop2))*pix*Sgrid;
% Myx1=sum(sum((gridY-y1).*Txcrop1))*pix*Sgrid;
% Myx2=sum(sum((gridY-y2).*Txcrop2))*pix*Sgrid;
% M1=[Mxx1 -Mxy1;-Myx1 Myy1];
% M2=[Mxx2 -Mxy2;-Myx2 Myy2];
% 
% %TBD
% Momentcell1(ff)=trace(M1);
% Momentcell2(ff)=trace(M2);
% disp(['Moment net contractile cellule 1: ',  num2str(Momentcell1(ff)),' N.m']);
% disp(['Moment net contractile cellule 2: ',  num2str(Momentcell2(ff)),' N.m']);

Forcex_cell1(ff)=sum(abs(Txcrop1(:)))*Sgrid;%sum of x component of forces (in magnitude) for cell 1
Forcey_cell1(ff)=sum(abs(Tycrop1(:)))*Sgrid;%sum of y component of forces (in magnitude) for cell 1
Forcex_cell2(ff)=sum(abs(Txcrop2(:)))*Sgrid;%sum of x component of forces (in magnitude) for cell 2
Forcey_cell2(ff)=sum(abs(Tycrop2(:)))*Sgrid;%sum of y component of forces (in magnitude) for cell 2
Deplcropx1=deplacementX.*BWgrid1;
Deplcropy1=deplacementY.*BWgrid1;
Deplcropx2=deplacementX.*BWgrid2;
Deplcropy2=deplacementY.*BWgrid2;
Deplx_cell1(ff)=sum(abs(Deplcropx1(:)));%sum of x component of displacements (in magnitude) for cell 1
Deply_cell1(ff)=sum(abs(Deplcropy1(:)));%sum of y component of displacements (in magnitude) for cell 1
Deplx_cell2(ff)=sum(abs(Deplcropx2(:)));%sum of x component of displacements (in magnitude) for cell 2
Deply_cell2(ff)=sum(abs(Deplcropy2(:)));%sum of y component of displacements (in magnitude) for cell 2


%% Force dipole
%Main contraction direction 

s=regionprops(mask_corr,'Area','Centroid');
[~,ind]=max([s.Area]);
 x0=s(ind).Centroid(:,1);
 y0=s(ind).Centroid(:,2);
Mxx=sum(sum((gridX-x0).*Txcrop))*pix*Sgrid^2;
Mxy=sum(sum((gridX-x0).*Tycrop))*pix*Sgrid^2;
Myx=sum(sum((gridY-y0).*Txcrop))*pix*Sgrid^2;
Myy=sum(sum((gridY-y0).*Tycrop))*pix*Sgrid^2;
M=[Mxx -Mxy;-Myx Myy];
Moment(ff)=trace(M);
[V,D]=eig(M);
if isreal(D)
    DD=max(-D);
%on identifie la direction principale (indice ind)
[Dmax, ind]=max(DD);
ang=180/pi*atan(real(V(2,ind))/real(V(1,ind)));
%angle par rapport la verticale
if ang>=0
Angle(ff)=90-ang;
else
Angle(ff)= -90-ang;
end
L1=D(ind,ind);
%autre direction
autre=mod(ind,2)+1;
L2=D(autre, autre);
else% non diagonalizable
    Mxyn=(Mxy+Myx)/2;
    Myxn=Mxyn;
    M=[Mxx -Mxyn;-Myxn Myy];
    [V,D]=eig(M);
DD=max(-D);
%on identifie la direction principale (indice ind)
[Dmax, ind]=max(DD);
ang=180/pi*atan(real(V(2,ind))/real(V(1,ind)));
%angle par rapport la verticale
if ang>=0
Angle(ff)=90-ang;
else
Angle(ff)= -90-ang;
end
L1=DD(ind);
%autre direction
autre=mod(ind,2)+1;
L2=DD(autre);
end
Polar(ff)=(L1-L2)/(L1+L2);
disp(['Angle of main contraction compared to the vertical: ',  num2str(Angle(ff))]);

disp(['Polarisation degree (0=isotropic,1=uniaxial): ',  num2str(Polar(ff))]);
%%
% 
Tmagn=sqrt(Tractionx.^2+Tractiony.^2);
figure(h3)
[L,C]=size(bf);
contourf(gridX,L-gridY,Tmagn),colormap(jet),colorbar
axis equal
set(gca,'Visible','off');
set(gca, 'clim', [0 2000]);
hold on
[c,h]=contour(gridX,L-gridY,BWgrid,[0.5 0.5]);
set(h,'LineWidth',1,'LineColor','w')
hold on
h=plot([130 130+10e-6/(interval*pix)], [15 15],'w','LineWidth',2);
%text(130,20,'10 um','Color','w')
title( ['Frame: ',num2str(ff)]);
hold off
% 
% figurepath=cat(2,path,'\figureTFM\Stress_magnitude',num2str(ff),'.tif');
% print(h3,fullfile(figurepath),'-dtiff','-r100');
% A=imread(figurepath);
% imwrite(A,cat(2,path,'\figureTFM\Stress_magnitude.tif'),'WriteMode','append');
% delete (figurepath)

%
figure(h2) % Forces on BF image
imshow(contrast(bf,0.1,0.1),[],'InitialMagnification','fit')
title(['Frame ', num2str(ff), '- Principal axis: green - Ellipse: polarisation degree'])
hold on
quiver(gridX(1:2:end,1:2:end),gridY(1:2:end,1:2:end),2*Txcrop(1:2:end,1:2:end),2*Tycrop(1:2:end,1:2:end),2,'r','AutoScale','off')
%  hold on
%  plot([x0-200*cos(pi/180*ang) x0+200*cos(pi/180*ang)],[y0+200*sin(pi/180*ang) y0-200*sin(pi/180*ang)],'g','LineWidth',2)
%  plot([x0-200*cos(pi/180*90)  x0+200*cos(pi/180*90)],[y0+200*sin(pi/180*90)  y0-200*sin(pi/180*90)],'k','LineWidth',2)
%  hold on
%  theta=linspace(0,2*pi);
%  x=100*cos(theta);
%  y=(1-Polar(ff))/(1+Polar(ff))*100*sin(theta);
%  xrot=x0+x*cos(pi/180*ang)+y*sin(pi/180*ang);
%  yrot=y0-x*sin(pi/180*ang)+y*cos(pi/180*ang);
%  plot(xrot,yrot,'m')
% hold off
% drawnow

figurepath=cat(2,path,'\figureTFM\BF_polar_degree',num2str(ff),'.tif');
print(h2,fullfile(figurepath),'-dtiff','-r100');
A=imread(figurepath);
imwrite(A,cat(2,path,'\figureTFM\BF_polar_degree.tif'),'WriteMode','append');
delete (figurepath)
%%
%save (fullfile(path,[filename(1:end-4),'_results', num2str(ff),'.mat']),'p_corr','BW','BWgrid','deplacement*','Traction*','eigenvalue*');
Dx(:,:,ff)=deplacementX;
Dy(:,:,ff)=deplacementY;
Tx(:,:,ff)=Tractionx;
Ty(:,:,ff)=Tractiony;

%% Je sauve les figures - faites avant en haute resolution (Cecile)

% print(h2,fullfile(path,['cell',num2str(ff),'_cell&force.tif']),'-dtiff'); %,'-r600'
% print(h3,fullfile(path,['cell',num2str(ff),'_stress.tif']),'-dtiff','-r600');
% print(h4,fullfile(path,['cell',num2str(ff),'_displacement.tif']),'-dtiff','-r600');
% print(h5,fullfile(path,['cell',num2str(ff),'_strain.tif']),'-dtiff','-r600');
% print(h6,fullfile(path,['cell',num2str(ff),'_pressure.tif']),'-dtiff','-r600');

end
save (fullfile(path,'Allresults2.mat'),'Brightfield','Mask_left','Mask_right','Dx','Tx','Dy','Ty','grid*');
% print(h6,fullfile(path,'displacement_profiles.tif'),'-dtiff','-r600');
% print(h7,fullfile(path,'strain_profiles.tif'),'-dtiff','-r600');

%graphs
pos=(1:Nb_frames);
graphs=figure('units','Normalized','Position', [0.01,0.03,0.99,0.9],'Name','Doublet de cellules');
subplot(2,2,1),plot(pos,U,'o-'),title('Strain Energy (J)');
subplot(2,2,2),plot(pos,Pmoy,'o-'),title('Average Stress (Pa)');
subplot(2,2,3),plot(pos,Dmoy,'d-g'),title('Average displacement (m)');
subplot(2,2,4), plot(pos,Angle, '--o'),title('Angle');

print(graphs,cat(2,path,'graphs1'),'-dpng');

graphs=figure('units','Normalized','Position', [0.01,0.03,0.99,0.9],'Name','Force Cellule-Cellule') ;
subplot(2,3,1), plot(pos,Cell_ECMratio1*100,'-dg'), hold on, plot(pos,Cell_ECMratio2*100,'-dm'),hold off, title('Rapport Fcell-cell/Fcell-ECM en %')
subplot(2,3,2), plot(pos,Forcey_cell1,'-om'), hold on, plot(pos,Forcey_cell2,'-dm'), legend('Fy cell1','Fy cell2'),title('Sum of y-Forces')
subplot(2,3,3), plot(pos,Fcellcell,'-om'), hold on, plot(pos,DFcellcell,'-dm'), legend('F cell-cell','DF cell-cell'),title('Intercellular Forces')
subplot(2,3,4), plot(pos,Polar,'-sc'),  title('Degree of polarisation')
subplot(2,3,5), plot(pos,Fvect,'-sc'),  title('Total vectorial sum of forces (N)')
subplot(2,3,6), plot(pos,Fvect./Ftot.*100,'-sc'),  title('out of equilibrium percentage')
print(graphs,cat(2,path,'graphs2'),'-dpng');

%Save results
fid1=fopen(fullfile(path,'results.txt'),'w+');
    fprintf(fid1,'Image  \t Ec \t Pmax \t Pmoy \t Dmoy \t Fvect \t DispMax \t StrainMax  \t Moment \t Angle \t Polar \t Cell/ECM_cell1 \t Cell/ECM_cell2 \t Forcey_cell1 \t Forcey_cell2 \t Fcell_cell \t DFcell_cell \n');
    fclose(fid1);
Mat=[(1:Nb_frames);U;Pmax;Pmoy;Dmoy;Fvect;DispMax;StrainMax;Moment;Angle;Polar;Cell_ECMratio1.*100;Cell_ECMratio2.*100;Forcey_cell1;Forcey_cell2;Fcellcell;DFcellcell];
dlmwrite(fullfile(path,'results.txt'),Mat','-append','delimiter','\t');
disp('Results saved in: ')
disp(fullfile(path,'results.txt'))


save (fullfile(path,[filename(1:end-4),'_param','.mat']),'grid*','interval','pas');
fid=fopen(fullfile(path,[filename,'_Param.txt']),'w');
fprintf(fid,'Registration (x direction): %g \n',regx);
fprintf(fid,'Registration (y direction): %g \n',regy);
fprintf(fid,'PIV window size: %g \n \n',window);
fprintf(fid,'Number of PIV iterations (added to the 1st): %g \n \n',iter);
fprintf(fid,'Overlap PIV: %g \n \n',overlapPIV);
fprintf(fid,'Overlap tracking: %g \n \n',overlapTrack);

fprintf(fid,'Tracking parameters: \n');
fprintf(fid,'Feature size: %g \n',featsize);
fprintf(fid,'Min intensity(masscut): %g\n',masscut);
fprintf(fid,'Max radius: %g\n',barrg);
fprintf(fid,'Max eccentricity: %g\n',barcc);
fprintf(fid,'Min ratio intensity/radius: %4.1f\n',IdivRg);
fprintf(fid,'Max displacement in pixels: %g\n\n',maxd);
fprintf(fid,'Pixel size (in meter): %g\n',pix);
fprintf(fid,'Data interval (for displacement and force) in pixels: %g\n',interval);
fprintf(fid,'Young modulus (in Pa): %g\n',E);
fprintf(fid,'Poisson ratio: %g\n',nu);
fprintf(fid,'Regularization parameter: %g\n',alphadef);
fclose(fid);

% make movies: 
% if movies
% writerObj = VideoWriter(fullfile(path,'cell_forces.avi'));
% writerObj.FrameRate = 8;
% open(writerObj);
% 
% mov = avifile(fullfile(path,'cell_forces.avi'),'compression','none','fps',8);
% h2=figure('units','Normalized','position',[0 0.04 1 0.88],'NumberTitle','off','Name','Phase+fleches');
% for i=1:Nb_frames
%     s=regionprops(Mask(:,:,i),'Centroid');
%  x0=s.Centroid(:,1);
%  y0=s.Centroid(:,2);
% figure(h2)
% imshow(Brightfield(:,:,i),[],'InitialMagnification','fit')
% axis equal
% set(gca,'Visible','off');
% title(['Image: ',num2str(i)]);
% hold on
% quiver(gridX,gridY,Tx(:,:,i), Ty(:,:,i),2,'r')
% hold off
% 
%  frame=getframe;
%          writeVideo(writerObj,frame);
% F(i)=getframe(gca,[5,5,1150,800]);
% mov = addframe(mov,F(i));
% end
% close(writerObj);
% clear writerObj
% 
% 
% writerObj = VideoWriter(fullfile(path,'forces_amplitude.avi'));
% writerObj.FrameRate = 8;
% open(writerObj);
% 
% h3=figure('units','Normalized','position',[0 0.04 1 0.88],'NumberTitle','off','Name','Amplitude traction');
% 
% figurepath=cat(2,path,'\figureTFM\Forces',num2str(ff),'.tif');
% print(h3,fullfile(figurepath),'-dtiff','-r100');
% A=imread(figurepath);
% imwrite(A,cat(2,path,'\figureTFM\Forces.tif'),'WriteMode','append');
% delete (figurepath)
% 
% for i=1:Nb_frames
%     
% BW=logical(Mask(:,:,i));
% BWgrid=logical(BW(gridY(:,1),gridX(1,:)));
% [L,C]=size(BW);
% figure(h3)
% Tmagn=sqrt(Tx(:,:,i).^2+Ty(:,:,i).^2);
%  contourf(gridX,L-gridY,Tmagn,linspace(0,max_stress,10)),caxis([0 max_stress]),colormap(jet)%,colorbar
% imagesc(Tmagn,[0 300]),colormap(jet),colorbar
% axis equal
% title(['Image: ',num2str(i)]);
% set(gca,'Visible','off');
% hold on
% [c,h]=contour(gridX,L-gridY,BWgrid,[0.5 0.5]);
% set(h,'LineWidth',1,'LineColor','w')
% hold on
% quiver(gridX,L-gridY,Tx(:,:,i), -Ty(:,:,i),2,'w')
% hold off
% 
% 
% if i==1
% 
%  frame=getframe;
%  [d1,d2,~]=size(frame.cdata);
% writeVideo(writerObj,frame);
% else
% frame=getframe(gca,[1 1 d2,d1]);
% writeVideo(writerObj,frame);
% end
% end
% close(writerObj);
% clear writerObj
% end
% hcolor=figure;
% contourf(gridX,L-gridY,Tmagn,linspace(0,maxP,10)),caxis([0 maxP]),colormap(jet)%,colorbar
% colorbar
% saveas(hcolor,fullfile(path,'scale.fig'))
% 
end
