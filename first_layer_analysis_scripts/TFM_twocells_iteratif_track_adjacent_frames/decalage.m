function [deltax,deltay,Cmax]=decalage(im1,im2)
%Author: Irene Wang
%Attention!! la taille des 2 images doivent �tre identiques
Nx=size(im1,2);
Ny=size(im1,1);
if rem(Nx,2)
    Nx=Nx-1;
    im1=im1(1:Ny,1:Nx);
    im2=im2(1:Ny,1:Nx);   
end
if rem(Ny,2)
     Ny=Ny-1;
     im1=im1(1:Ny,1:Nx);
    im2=im2(1:Ny,1:Nx);
end

A1=real(ifft2(fft2(im1) .* fft2(rot90(im1,2))));
A2=real(ifft2(fft2(im2) .* fft2(rot90(im2,2))));
M1=max(A1(:));
M2=max(A2(:));
C = real(ifft2(fft2(im1) .* fft2(rot90(im2,2))));
C2=fftshift(C);
%figure, imshow(C2,[])
[Vmax,imax]=max(C2(:));
Cmax=Vmax/sqrt(M1*M2);
[I,J]=ind2sub(size(C),imax);
if (I==size(C,1)) || (I==1)
    Isub=I;
else
%mesure d�calage pr�cision subpixel
    Isub=I+(log(C2(I-1,J))-log(C2(I+1,J)))/(2*(log(C2(I-1,J))+log(C2(I+1,J))-2*log(C2(I,J))));
end
if (J==size(C,2)) || (J==1)
    Jsub=J;
else
Jsub=J+(log(C2(I,J-1))-log(C2(I,J+1)))/(2*(log(C2(I,J-1))+log(C2(I,J+1))-2*log(C2(I,J))));
end
deltax=Jsub-Nx/2; 
deltay=Isub-Ny/2;
% si deltay>0 im2 d�cal�e vers le haut par rapport � im1
% si deltax>0 im2 d�cal�e vers la gauche par rapport � im1