function [ panoImg ] = imageStitching(img1, img2, H2to1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

size1 = size(img1);
size2 = size(img2);

%Warp the image allowing for a lot of movement
img2_warp = warpH(img2,H2to1,2*size2);

%convert both images to double
img2_warp = im2double(img2_warp);
img1 = im2double(img1);

%Pad the first image to be the same size as image 2
padsize = size(img2_warp)-size(img1);
img1 = padarray(img1,padsize,0,'post');

%take the maximum value pixel for each image at any location, this helps
%deal with zero locations and the overlap well
panoImg = max(img1,img2_warp);
%convert floats back to an image
panoImg = uint8(floor(256*panoImg));
end

