function [locsDoG] = getLocalExtrema(DoGPyramid, DoGLevels, ...
    PrincipleCurvature, th_contrast, th_r)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here


%create a mask of the region to look for local extrema
max_region = zeros(3,3,3);
max_region(:,:,1) = [0,0,0;0,1,0;0,0,0];
max_region(:,:,2) = [1,1,1;1,1,1;1,1,1];
max_region(:,:,3) = [0,0,0;0,1,0;0,0,0];
%find all local extrema
logical_max = imregionalmax(DoGPyramid,max_region);

%filter out high principle curvature or low contrast from the maxima
DoGPyramid_thresh = DoGPyramid > th_contrast;
PrincipleCurvature_thresh = PrincipleCurvature < th_r;
thresh = DoGPyramid_thresh .* PrincipleCurvature_thresh;

logical_max = logical_max.*thresh;

%convert the location of the maxima to x y coordinates
[y,x,levels] = ind2sub(size(logical_max),find(logical_max));
locsDoG = [x,y,DoGLevels(levels)'];


end

