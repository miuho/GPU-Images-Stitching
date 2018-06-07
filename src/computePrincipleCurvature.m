function [principleCurvature] = computePrincipleCurvature(DoGPyramid)


principleCurvature = zeros(size(DoGPyramid));
for i = 1:size(DoGPyramid,3);
    %Calculate the second derivatives of the DoG pyramid
    [Dx,Dy] = gradient(DoGPyramid(:,:,i));
    [Dxx,Dxy] = gradient(Dx);
    [Dyx,Dyy] = gradient(Dy);
    %calculate the principle curvature based on the hermitian for every
    %point
    principleCurvature(:,:,i) = ((Dxx+Dyy).^2)./((Dxx.*Dyy)-Dxy.*Dyx);
    
end

