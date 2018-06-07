function [locsDoG, GaussianPyramid] = DoGdetector( im, sigma0, k, levels,...
    th_contrast,th_r )

%Create pyramid based on gaussian filtered images
GaussianPyramid = createGaussianPyramid(im,sigma0,k,levels);
%Create differnce of gaussian pyramid to find interest points
[DoGPyramid, DoGLevels] = createDoGPyramid(GaussianPyramid,levels);
%calculate principle curvature to help filter edges
PrincipleCurvature = computePrincipleCurvature(DoGPyramid);
%find the local maximum in difference of gaussian while ignoring edges
%and dark patches
locsDoG = getLocalExtrema(DoGPyramid,DoGLevels,PrincipleCurvature,...
    th_contrast,th_r);

end

