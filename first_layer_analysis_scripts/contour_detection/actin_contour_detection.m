%%
%Author: Irene Wang
% description of the script: First, intensity profiles along the y-axis are
% calculated. w lines are averaged together to calculate those profiles and
% then smoothed with a moving window of size span.
%V2 changes on 19/01/21:
%-segmentation removed
%- if not sure, value is ignored (=nan)
function [Xtopf, Ytopf, Xbottomf, Ybottomf] = actin_contour_detection(I,corners)
pixsize=0.108;%micron (to guess the slope)
% N_profiles=40;%number of vertical profiles
w=20;%averaging width in pixels (for calculating the profiles)
step = 10; %distance in pixel between the profiles
ymargin=10;%nb of pixels outwards added to the bounding box in y direction for profile
yinsearch=60;%distance inward from the cell edge to search for fiber
span = 10; % Size of the averaging window for smoothing the profiles
%parameters for edge detection
rint_min=0.1;%ratio of low intensity threshold to total span
rint_max=0.9;%ratio of high intensity threshold to total span
rslope=0.2;%slope detection coefficient (fraction of the total span increase for psf width)
%parameters for filtering
angle_max=10;%angle max (en degrés) entre 2 segments pour filtrage spatial
%angle_edge=10;%angle max (en degrés) pour le point du bord
time_window=10;%in frames, to remove outliers in temporal filtering
time_span=10;%in frames, for median filtering in time
display=false;%set this true if you want to see the plots of the intermediate steps

N_images = size(I,3);

%  x_sort = sort(corners(1,:));
%  y_sort = sort(corners(2,:));
%
% xstart = x_sort(1);
% xwidth = x_sort(4)-x_sort(1);
% ystart = y_sort(1);
% ywidth = y_sort(4)-y_sort(1);

xstart = min(corners(1,1),corners(1,4));
xwidth = max(corners(1,2),corners(1,3))-xstart;
ystart =  min(corners(2,1),corners(2,2));
ywidth = max(corners(2,3),corners(2,4))-ystart;


figure, imshow(I(:,:,1),[]);
roi=impoly(gca,corners');
mask=createMask(roi);
%imshow(mask);

xpos = xstart+step:step:(xstart+xwidth)-step;
N_profiles = size(xpos,2);

%output arrays
Xtop=zeros(N_profiles,N_images);
Xbottom=zeros(N_profiles,N_images);
Ytop=zeros(N_profiles,N_images);
Ybottom=zeros(N_profiles,N_images);
%%
for i=1:N_images
    It=I(:,:,i);
    
    
    %%
    if display
        figure(15),
        imshow(I(:,:,i),[]), hold on
        [c,h]=contour(mask,[0.5 0.5]);
        set(h,'LineWidth',1,'LineColor','g')
        hold on
        plot([xstart xstart xstart+xwidth xstart+xwidth xstart],[ystart ystart+ywidth ystart+ywidth ystart ystart],'r')
        hold on
    end
    
    % profil positions
    % step=(xwidth-2*xinmargin)/(N_profiles-1);
    
    Xtop(:,i)=xpos;
    Xbottom(:,i)=xpos;
    y1=round(ystart-ymargin);
    y2=round(ystart+ywidth+ymargin);
    profile=zeros(N_profiles,y2-y1+1);
    
    
    
    for j=1:N_profiles
        profile(j,:)=mean(It(y1:y2,round(xpos(j)-w/2):round(xpos(j)+w/2)),2);
        maskprof=mask(y1:y2,round(xpos(j)));
        y1start=max(1,find(maskprof,1,'first')-ymargin);
        y2fin=min(length(maskprof),find(maskprof,1,'last')+ymargin);
        y1fin=find(maskprof,1,'first')+yinsearch;
        y2start=find(maskprof,1,'last')-yinsearch;
        if display
            imshow(It);hold on;
            plot([xpos(j) xpos(j)],[y1start+y1-1 y1fin+y1-1],'y'), hold on
            plot([xpos(j) xpos(j)],[y2start+y1-1 y2fin+y1-1],'y'), hold on
        end
        prof1=profile(j,y1start:y1fin);
        prof2=profile(j,y2start:y2fin);
        
        w = ones(span,1)/span;
        %traitement 1ere moitie du  profil
        temp= conv(prof1,w,'same');
        
        if size(temp,2)-span <0
            lol=1;
        end
        
        prof1s=temp(span:end-span);
        pent1=diff([prof1s prof1s(end)]);
        %definition de seuils pour intensite et pente
        int_min=min(prof1s)+(max(prof1s)-min(prof1s))*rint_min;
        int_max=min(prof1s)+(max(prof1s)-min(prof1s))*rint_max;
        slope_thres=rslope*(max(prof1s)-min(prof1s))/(0.5/pixsize);
        IND=find((pent1>slope_thres)&(prof1s>int_min)&(prof1s<int_max));
        %selection du meilleur indice si segment contigu
        if ~isempty(IND)
            dIND=[diff(IND) 0];
            if ~isempty(find(dIND==1, 1))
                imin=find(dIND==1, 1,'first');
                %[INDmin,imin]=min(IND(dIND==1));
                imax=imin;
                while dIND(imax)==1
                    imax=imax+1;
                end
                [~,INDselect]=max(prof1s(IND(imin):IND(imax)));
                x1=IND(imin)+INDselect-1+y1start+span-1;
            else
                x1=nan;
                %x1=min(IND)+y1start+span-1;%celui qui est sur le bord
            end
        else
            %si pas de segment contigu: on marque qu'il y a un pb
            x1=nan;
            %x1=find(maskprof,1,'first');
        end
        Ytop(j,i)=x1+y1-1;
        
        %traitement 2eme moitie du  profil
        temp= conv(prof2,w,'same');
        prof2s=temp(span:end-span);
        pent2=diff([prof2s prof2s(end)]);
        %definition de seuil pour intensite et pente
        int_min=min(prof2s)+(max(prof2s)-min(prof2s))*rint_min;
        int_max=min(prof2s)+(max(prof2s)-min(prof2s))*rint_max;
        slope_thres=-rslope*(max(prof2s)-min(prof2s))/(0.5/pixsize);
        IND=find((pent2<slope_thres)&(prof2s>int_min)&(prof2s<int_max));
        %selection d'un segment contigu
        if ~isempty(IND)
            dIND=[0 diff(IND)];
            if ~isempty(find(dIND==1, 1))
                imax=find(dIND==1,1, 'last');
                imin=imax;
                while dIND(imin)==1
                    imin=imin-1;
                end
                [~,INDselect]=max(prof2s(IND(imin):IND(imax)));
                x2=IND(imin)+INDselect-1+y2start+span-1;
            else
                x2=nan;
                %x2=max(IND)+y2start+span-1;
            end
        else
            x2=nan;
            %x2=find(maskprof,1,'last');
        end
        Ybottom(j,i)=x2+y1-1;
        % figure(20)
        % plot(profile(j,:))
        % hold on, plot([x1 x2],[profile(j,x1) profile(j,x2)],'ro'), hold off
        % title(num2str(j))
        % drawnow
        % waitforbuttonpress
        if display
            plot([xpos(j) xpos(j)],[Ytop(j,i) Ybottom(j,i)],'oc'), hold on
            drawnow
        end
    end
    if display
        hold off
    end
end
%% filtrage des valeurs obtenues

%filtered arrays
Xtopf=Xtop;
Xbottomf=Xbottom;
Ytopf=Ytop;
Ybottomf=Ybottom;

% filtrage temporel
for j=1:N_profiles
    %upper fiber
    %   figure(10),plot(Yupper(j,:),'og'), hold on
    % remove outliers in time windows
    nsegment=ceil(2*N_images/time_window);
    jstart=linspace(1,N_images-time_window+1,nsegment);
    for iter=1:4 % iterations
        y=Ytop(j,:);
        t=(1:N_images);
        ind_to_remove=find(isnan(y));
        for jj=1:nsegment
            segment=y(round(jstart(jj)):round(jstart(jj))+time_window-1);
            var_permis=max(std(segment),5);
            IND=find((segment>(median(segment)+2.5*var_permis))|(segment<(median(segment)-1.5*var_permis)));
            ind_to_remove=[ind_to_remove IND+round(jstart(jj))-1];
        end
        ind_to_remove=unique(ind_to_remove);
        y(ind_to_remove)=[];
        t(ind_to_remove)=[];
        y2=interp1(t,y,(1:N_images));
        if isnan(y2(1))
            debut=find(~isnan(y2),1,'first');
            y2(1:debut-1)=y2(debut);
        end
        if isnan(y2(end))
            fin=find(~isnan(y2),1,'last');
            y2(fin+1:end)=y2(fin);
        end
        Ytopf(j,:)=y2;
    end
    %median filtering
    ys=medfilt(Ytopf(j,:),time_span);
    Ytopf(j,:)=ys;
    %plot(Yupper(j,:),'+r'), hold off
    %waitforbuttonpress
    %bottom fiber
    %figure(10),plot(Yupper(j,:),'og'), hold on
    % remove outliers in time windows
    nsegment=ceil(2*N_images/time_window);
    jstart=linspace(1,N_images-time_window+1,nsegment);
    for iter=1:4 % iterations
        y=Ybottom(j,:);
        t=(1:N_images);
        ind_to_remove=find(isnan(y));
        for jj=1:nsegment
            segment=y(round(jstart(jj)):round(jstart(jj))+time_window-1);
            var_permis=max(std(segment),5);
            IND=find((segment>(median(segment)+1.5*var_permis))|(segment<(median(segment)-2.5*var_permis)));
            ind_to_remove=[ind_to_remove IND+round(jstart(jj))-1];
        end
        ind_to_remove=unique(ind_to_remove);
        y(ind_to_remove)=[];
        t(ind_to_remove)=[];
        y2=interp1(t,y,(1:N_images));
        if isnan(y2(1))
            debut=find(~isnan(y2),1,'first');
            y2(1:debut-1)=y2(debut);
        end
        if isnan(y2(end))
            fin=find(~isnan(y2),1,'last');
            y2(fin+1:end)=y2(fin);
        end
        Ybottomf(j,:)=y2;
    end
    %median filtering
    ys=medfilt(Ybottomf(j,:),time_span);
    Ybottomf(j,:)=ys;
    %plot(Ybottomf(j,:),'+r'), hold off
    %waitforbuttonpress
end

%% filtrage spatial
%remove points outside of the cornerpoints
INDtop = find(Xtopf(:,1)<corners(1,1)|Xtopf(:,1)>corners(1,2));
INDbottom = find(Xbottomf(:,1)<corners(1,4)|Xbottomf(:,1)>corners(1,3));

Xtopf(INDtop,:)=[];
Ytopf(INDtop,:)=[];
Xbottomf(INDbottom,:)=[];
Ybottomf(INDbottom,:)=[];

% put cornerpoints as fixpoints on all timeframes
Xtopf(1,:) = corners(1,1);
Xtopf(end,:) = corners(1,2);
Xbottomf(end,:) = corners(1,3);
Xbottomf(1,:) = corners(1,4);
Ytopf(1,:) = corners(2,1);
Ytopf(end,:) = corners(2,2);
Ybottomf(end,:) = corners(2,3);
Ybottomf(1,:) = corners(2,4);

for i=1:N_images
    if display
        figure(25),
        imshow(I(:,:,i),[]), hold on
        plot(Xtop(:,i) ,Ytop(:,i) ,'oc'), hold on
        plot(Xtopf(:,i) ,Ytopf(:,i) ,'+g'), hold on
        plot(Xbottom(:,i) ,Ybottom(:,i) ,'oc'), hold on
        plot(Xbottomf(:,i) ,Ybottomf(:,i) ,'+g'), hold on
    end
    %remove outliers
    y=Ytopf(:,i);
    x=Xtopf(:,i);
    IND=find((y>(median(y)+3*std(y)))|(y<(median(y)-3*std(y))));
    IND(IND==1)=[];
    IND(IND==numel(y))=[];
    y(IND)=[];
    x(IND)=[];
    Ytopf(:,i)=interp1(x,y,Xtopf(:,i));
    %angles
    vecteurs1=[(Xtopf(2:end,i)-Xtopf(1:end-1,i)),(Ytopf(2:end,i)-Ytopf(1:end-1,i))];
    pscal=sum(vecteurs1(1:end-1,:).*vecteurs1(2:end,:),2)./(sqrt(sum(vecteurs1(1:end-1,:).^2,2)).*sqrt(sum(vecteurs1(2:end,:).^2,2)));
    pscal(pscal>1)=1;
    pscal(pscal<-1)=-1;
    angles1=180/pi*acos(pscal);
    %pour les bords: si le premier angle est ok, on garde le premier point/
    %idem pour le dernier point (attention, longueur des vecteurs changés)
    y=Ytopf(:,i);
    x=Xtopf(:,i);
    % if angles1(1)>angle_edge
    %     x(1)=[];
    %     y(1)=[];
    %     angles1(1)=[];
    % %     Xupper(1,i)=nan;
    % %     Yupper(1,i)=nan;
    % %     angles1(1)=nan;
    % end
    % if angles1(end)>angle_edge
    %     x(end)=[];
    %     y(end)=[];
    %     angles1(end)=[];
    % end
    IND=find(angles1>angle_max);
    IND2 = find(angles1<0);
    IND = [IND, IND2];
    if numel(IND)>1
        [A,is]=sort(angles1(IND),'descend');
        m=fix(numel(IND)/2);
        y(IND(is(1:m))+1)=[];
        x(IND(is(1:m))+1)=[];
        Ytopf(:,i)=interp1(x,y,Xtopf(:,i));
        y=Ytopf(:,i);
        x=Xtopf(:,i);
        x(isnan(y))=[];
        y(isnan(y))=[];
        vecteurs2=[(x(2:end)-x(1:end-1)),(y(2:end)-y(1:end-1))];
        pscal=sum(vecteurs2(1:end-1,:).*vecteurs2(2:end,:),2)./(sqrt(sum(vecteurs2(1:end-1,:).^2,2)).*sqrt(sum(vecteurs2(2:end,:).^2,2)));
        pscal(pscal>1)=1;
        pscal(pscal<-1)=-1;
        angles2=180/pi*acos(pscal);
        IND=find(angles2>angle_max);
        y(IND+1)=[];
        x(IND+1)=[];
    else
        y(IND+1)=[];
        x(IND+1)=[];
    end
    Ytopf(:,i)=interp1(x,y,Xtopf(:,i));
    
    %bottom profile
    %remove outliers
    y=Ybottomf(:,i);
    x=Xbottomf(:,i);
    IND=find((y>(median(y)+3*std(y)))|(y<(median(y)-3*std(y))));
    IND(IND==1)=[];
    IND(IND==numel(y))=[];
    y(IND)=[];
    x(IND)=[];
    Ybottomf(:,i)=interp1(x,y,Xbottomf(:,i));
    %angle
    vecteurs2=[(Xbottomf(2:end,i)-Xbottomf(1:end-1,i)),(Ybottomf(2:end,i)-Ybottomf(1:end-1,i))];
    pscal=sum(vecteurs2(1:end-1,:).*vecteurs2(2:end,:),2)./(sqrt(sum(vecteurs2(1:end-1,:).^2,2)).*sqrt(sum(vecteurs2(2:end,:).^2,2)));
    pscal(pscal>1)=1;
    pscal(pscal<-1)=-1;
    angles2=180/pi*acos(pscal);
    %pour les bords: si le premier angle est ok, on garde le premier point/
    %idem pour le dernier point (attention, longueur des vecteurs changés)
    y=Ybottomf(:,i);
    x=Xbottomf(:,i);
    % if angles2(1)>angle_edge
    %     x(1)=[];
    %     y(1)=[];
    %     angles2(1)=[];
    % end
    % if angles2(end)>angle_edge
    %     x(end)=[];
    %     y(end)=[];
    %     angles2(end)=[];
    % end
    IND=find(angles2>angle_max);
    
    if numel(IND)>1
        [A,is]=sort(angles2(IND),'descend');
        m=fix(numel(IND)/2);
        y(IND(is(1:m))+1)=[];
        x(IND(is(1:m))+1)=[];
        Ybottomf(:,i)=interp1(x,y,Xbottomf(:,i));
        y=Ybottomf(:,i);
        x=Xbottomf(:,i);
        x(isnan(y))=[];
        y(isnan(y))=[];
        vecteurs2=[(x(2:end)-x(1:end-1)),(y(2:end)-y(1:end-1))];
        pscal=sum(vecteurs2(1:end-1,:).*vecteurs2(2:end,:),2)./(sqrt(sum(vecteurs2(1:end-1,:).^2,2)).*sqrt(sum(vecteurs2(2:end,:).^2,2)));
        pscal(pscal>1)=1;
        pscal(pscal<-1)=-1;
        angles2=180/pi*acos(pscal);
        IND=find(angles2>angle_max);
        y(IND+1)=[];
        x(IND+1)=[];
    else
        y(IND+1)=[];
        x(IND+1)=[];
    end
    Ybottomf(:,i)=interp1(x,y,Xbottomf(:,i));
    
    if display
        plot(Xtopf(:,i) ,Ytopf(:,i) ,'r'), hold on
        plot(Xbottomf(:,i) ,Ybottomf(:,i) ,'r'), hold off
        %waitforbuttonpress
    end
end
%% final display
ll=figure('units','Normalized','position',[0.02 0.05 0.9 0.83],'Name','Tracking');

% if ~exist(fullfile(path,'actincontour'),'dir')
%     mkdir(fullfile(path,'actincontour'))
% end
% for i=1:N_images
%     figure(11);
%     imshow(I(:,:,i),[]), hold on, title(['Frame ',num2str(i)])
%     plot(Xupperf(:,i) ,Yupperf(:,i) ,'.-y','LineWidth',2), hold on
%     plot(Xbottomf(:,i) ,Ybottomf(:,i) ,'.-y','LineWidth',2), hold off
% %    plot(repmat(305,[1,1000]),1:1000 ,'.-y','LineWidth',2), hold on
% %    plot(repmat(690,[1,1000]),1:1000 ,'.-y','LineWidth',2), hold off
%     drawnow
% %     figurepath=cat(2,path,'actincontour\contour',num2str(i),'.tif');
% %     print(ll,fullfile(figurepath),'-dpng');
% %     A=imread(figurepath);
% %     imwrite(A,cat(2,path,'\actincontour\contour.tif'),'WriteMode','append');
% %     delete (figurepath)
%     %pause(0.1)
% end

% for i=1:N_images
%     Yup_ip(:,i) = interp1(Xupperf(:,i),Yupperf(:,i), X_ip);
%     Ybo_ip(:,i) = interp1(Xbottomf(:,i),Ybottomf(:,i), X_ip);
%     Yup_ip(:,i) = fill_nans(Yup_ip(:,i));
%     Ybo_ip(:,i) = fill_nans(Ybo_ip(:,i));
% end
% plot calculated contours
for i=1:N_images
    figure(11);
    imshow(I(:,:,i),[]), hold on, title(['Frame ',num2str(i)])
    plot(Xtopf(:,i) ,Ytopf(:,i) ,'.-y','LineWidth',2), hold on
    plot(Xbottomf(:,i) ,Ybottomf(:,i) ,'.-y','LineWidth',2), hold on
    drawnow
    %pause(0.5)
end
% % plot interpolated contours
% for i=1:N_images
%     figure(11);
%     imshow(I(:,:,i),[]), hold on, title(['Frame ',num2str(i)])
%     plot(X_ip ,Yup_ip(:,i) ,'.-y','LineWidth',2), hold on
%     plot(X_ip ,Ybo_ip(:,i) ,'.-y','LineWidth',2), hold on
%     plot(repmat(xstart+xinmargin/2,[1,1000]),1:1000 ,'.-y','LineWidth',2), hold on
%     plot(repmat(xstart+xwidth-xinmargin/2,[1,1000]),1:1000 ,'.-y','LineWidth',2), hold off
%     drawnow
% end


% dlmwrite(fullfile(path,[filename(1:end-4) '_Xup.txt']), Xupperf,'\t');
% dlmwrite(fullfile(path,[filename(1:end-4) '_Yup.txt']), Yupperf,'\t');
% dlmwrite(fullfile(path,[filename(1:end-4) '_Xbott.txt']), Xbottomf,'\t');
% dlmwrite(fullfile(path,[filename(1:end-4) '_Ybott.txt']), Ybottomf,'\t');
% %% saving masks
% if strcmp(save,'Yes')
%     for i=1:N_images
%         if i==1
%     imwrite(uint8(255*masks{i}),fullfile(path,[filename(1:end-4) '_mask.tif']));
%         else
%             imwrite(uint8(255*masks{i}),fullfile(path,[filename(1:end-4) '_mask.tif']),'WriteMode','append');
%         end
%     end
% end
end