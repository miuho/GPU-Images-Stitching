function [ locs, desc ] = computeBrief(im, GaussianPyramid, locsDoG, k, ...
    levels, compareX, compareY)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

patchWidth = 9;
edgeBoundry = (patchWidth + 1)/2;

[height, width,~] = size(im);

%remove locations near the edge
locs = locsDoG((locsDoG(:,1) >= edgeBoundry) & ...
    (locsDoG(:,2) >= edgeBoundry) & ...
    (locsDoG(:,1) <= width-edgeBoundry) &...
    (locsDoG(:,2) <= height-edgeBoundry), :);
[~,levels_idx] = ismember(locs(:,3),levels);

desc = [];
for i = 1:size(locs,1);
    %get the center of each interest point
    cur_y = locs(i,2);
    cur_x = locs(i,1);
    %extract the 9x9 patch from the image
    patch = GaussianPyramid(cur_y-4:cur_y+4,cur_x-4:cur_x+4,levels_idx(i));
    %get a descripter by linear index comparison
    desc = [desc;(patch(compareX)<patch(compareY))'];
end

