%Author: Irene Wang
function [S, all_angles] = Actin_analysis(I)
% dirname = uigetdir(path,'Choose directory with actin images');
%     dirname = path;
%     files = dir(dirname);
%     ind=cellfun(@find,{files(:).isdir},'UniformOutput',false); 
%     files=files(cellfun(@isempty,ind));
%     ind3=cellfun(@(s) strfind(s,'tif'),{files(:).name},'UniformOutput',false); 
%     files=files(~cellfun(@isempty,ind3));
%     Num_cell=length(files);
%     disp([num2str(Num_cell), ' images found in directory.'])  
%     
%     Sorder=zeros(1,Num_cell);%to store value of order parameter
%     MeanAng=zeros(1,Num_cell);%to store value of average orientation angle
    
    w_smooth=2;%width in pixels of gaussian smoothing of the image
    w_orientation=6;%Gaussian waist to define local neighbourhood
    cohe_thres=0.7;%threshold on coherency value for the orientation angle to be taken into account

    % I=imread(path);  
    [M,N]=size(I);
    %% Image smoothing
    I2=imadjust(I);
    [X,Y]=meshgrid((-5:5),(-5:5));
    G=exp(-2*(X.^2+Y.^2)/w_smooth^2);
    h=G/sum(G(:));
    I=imfilter(I2,h);
    %% detect region out of the cell
    %determine maximum level (1% des pixels les plus lumineux)
    Npix=numel(I);
    bins=linspace(0,max(I(:)),100);
    n=hist(I(:),bins);
    ncumul=n(end);
    rec=0;
    while ncumul<0.005*Npix
    rec=rec+1;
    ncumul=sum(n(end-rec:end));
    end
    Niveau_max=bins(end-rec-1);
    bins=linspace(0,Niveau_max,40);
    n=hist(I(:),bins);
    %je cherche le minimum de l'histogramme lissé
    span = 4; % Size of the averaging window
    w = ones(span,1)/span; 
    smoothed_n = conv(n,w,'same');
    penten=diff(smoothed_n);
    ind=find((penten(1:end-1).*penten(2:end)<0)&(penten(1:end-1)<0));
    th1=bins(ind(1));  % define segmentation threshold
    if (th1/Niveau_max<0.02)||(th1/Niveau_max>0.5)
    disp('Problème probable de détermination du seuil de segmentation!')
    th1=Niveau_max*0.1;%*0.04;%
    end
    hors_cell=I<th1;
    Bk=mean(I(hors_cell));% background
    mask1=~hors_cell;
    %clean cell mask
    CC=bwconncomp(mask1);
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    for kk=1:CC.NumObjects
        if kk~=idx
        mask1(CC.PixelIdxList{kk}) = 0;
        end
    end
    se=strel('disk',4);
    cell_mask=imerode(mask1,se);
    
    %display
    IRGB=zeros([size(I) 3]);
    Itemp=I/max(I(:));
    IRGB(:,:,1)=Itemp;
    IRGB(:,:,2)=Itemp;
    Itemp(~cell_mask)=1;
    IRGB(:,:,3)=Itemp;
%    hfig=figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%     subplot(1,2,1),imshow(I,[],'InitialMagnification','Fit'),title('Smoothed image');
%    subplot(1,2,1),imshow(IRGB,'InitialMagnification','Fit'),title('Cell segmentation');
%    subplot(1,2,2)
    %% structure tensor
    WinN=14;%size in pixels of the window over which the tensor will be calculated
    w=w_orientation;%Gaussian waist to define local neighbourhood
    [X,Y]=meshgrid((-WinN/2:WinN/2),(-WinN/2:WinN/2));
    G=exp(-2*(X.^2+Y.^2)/w^2);
    J=zeros([size(I) 3]);%J11,J12,J22
    ind_cell=find(cell_mask);
    tic
    for ii=1:numel(ind_cell)
        u=rem(ind_cell(ii),N);
        v=(ind_cell(ii)-u)/N+1;
        if (u-WinN/2-1)>0 && (v-WinN/2-1)>0 && (u+WinN/2+1)<=M && (v+WinN/2+1)<=N
%           [u1,v1]=ind2sub(size(I),ind_cell(ii));
%           disp([num2str(u) num2str(u1)]);
%           disp([num2str(v) num2str(v1)]);
            window=I(u-WinN/2-1:u+WinN/2+1,v-WinN/2-1:v+WinN/2+1);
            Dx=window(2:end-1,3:end)-window(2:end-1,1:end-2);%dérivée partielle
            Dy=window(3:end,2:end-1)-window(1:end-2,2:end-1);
            J(u,v,1)=sum(sum(G.*(Dx.^2)));%J11
            J(u,v,2)=sum(sum(G.*Dx.*Dy));%J12
            J(u,v,3)=sum(sum(G.*(Dy.^2)));%J22
        end
     end
    toc
    theta=atan2(2*J(:,:,2),(J(:,:,3)-J(:,:,1)))/2;% orientation between -pi/2 and pi/2
    int_var=J(:,:,3)+J(:,:,1);%constant level or change
    coherency=sqrt((J(:,:,3)-J(:,:,1)).^2+4*J(:,:,2).^2)./int_var;%confidence in anisotropy measurement
    
%     theta(theta==0) = [];
                        % show histogram of angle
    [counts, binlocation] = histcounts(theta,90); 
    
%     hfig=figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);

%     subplot(1,2,1),imshow(IRGB,'InitialMagnification','Fit'),title('Cell segmentation');
%     subplot(1,2,2), histogram(theta);
%     if ~exist(fullfile(path,'histograms'),'dir')
%         mkdir(fullfile(path,'histograms'))
%     end

%     figurepath=cat(2,path,'\histograms\segmhist',num2str(g),'.tif');
%     print(hfig,fullfile(figurepath),'-dtiff','-r100');
%     A=imread(figurepath);
%     imwrite(A,cat(2,path,'\histograms\segmhist.tif'),'WriteMode','append');
%     delete (figurepath);
% 
%     peakpos = peakseek(counts);
%     angles = 180/pi*mean(abs(theta));
    result_mask=coherency>cohe_thres;%threshold for coherency
    %% display
  
    Orient_hsv=zeros(M,N,3);
    Orient_hsv(:,:,1)=(theta+pi/2)/pi;
    Orient_hsv(:,:,2)=coherency;
    Orient_hsv(:,:,3)=int_var./(int_var+(Niveau_max/Bk)*3*var(I(hors_cell))).*result_mask;%I/max(I(:)).*result_mask;%
    Orient_rgb=hsv2rgb(Orient_hsv);
    Orient_hsv2=Orient_hsv;
    Orient_hsv2(:,:,3)=I/max(I(:));%.*result_mask;
    Orient_rgb2=hsv2rgb(Orient_hsv2);
%     ho=figure('Name','Actin orientation','Units','normalized','Position',[0.1 0.1 0.8 0.8]);
%     subplot(1,2,1),imshow(Orient_rgb,'InitialMagnification','Fit'), title('hue=angle - saturation=coherency - value~local contrast');
%     subplot(1,2,2),imshow(Orient_rgb2,'InitialMagnification','Fit'), title('hue=angle - saturation=coherency - value=original image');
     
     %% Order parameter
     %calculate over region where coherency is above threshold
     %1. average orientation of the cell
     theta_th=theta.*result_mask;
     indval=find(theta_th);
     all_angles = theta_th(indval);
     AngMoy = mean(theta_th(indval));
     disp(['Average orientation: ', num2str(AngMoy*180/pi), ' deg'])
     %2. Order parameter
     Simage=cos(2*(theta-AngMoy)).*result_mask;
     S=mean(cos(2*(theta_th(indval)-AngMoy)));
%      hS=figure('Name','Order_parameter');
%      imshow(Simage,[]),colormap(jet), colorbar
%      hold on
%      htemp=plot([N/2-200*cos(AngMoy) N/2+200*cos(AngMoy)],[M/2+200*sin(AngMoy) M/2-200*sin(AngMoy)],'k');
%      set(htemp,'LineWidth',2)
%      hold off
%      disp(['Order parameter: ', num2str(S)])
%      imshow(Simage,[]),colormap(jet), colorbar
%      hold on
%      htemp=plot([N/2-200*cos(AngMoy) N/2+200*cos(AngMoy)],[M/2+200*sin(AngMoy) M/2-200*sin(AngMoy)],'k');
%      set(htemp,'LineWidth',2)
%      hold off
%      disp(['Order parameter: ', num2str(S)])
     
%      Sorder(i)=S;
%      MeanAng(i)=AngMoy*180/pi;
%      %% Save figure and color image
%      name=files(i).name;
%      saveas(ho,fullfile(path,[name(1:end-4),'_orientation.fig']));
%      imwrite(Orient_rgb, fullfile(path,[name(1:end-4),'_color.jpg']),'jpg','Quality',95);
%      imwrite(Orient_rgb2, fullfile(path,[name(1:end-4),'_color2.jpg']),'jpg','Quality',95);
%      saveas(hS,fullfile(path,[name(1:end-4),'_order.fig']));
% % save results
% fid=fopen(fullfile(path,'Results.txt'),'w');
% fprintf(fid,'Image name \t Mean angle (deg) \t   Order parameter\n');
% fclose(fid);
% 
% fid=fopen(fullfile(path,'Results.txt'),'a');
% for i=1:Num_cell
%     fprintf(fid,[files(i).name, '\t %g \t %g \n'],MeanAng(i), Sorder(i));
% end
% fclose(fid);
end