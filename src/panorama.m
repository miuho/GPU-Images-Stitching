clear;
%%{
scale = [512,1024];
im1 = imread('incline_L.png');
im2 = imread('incline_R.png');
im1 = imresize(im1,scale);
im2 = imresize(im2,scale);
locs1 = csvread('image1.txt');
locs2 = csvread('image2.txt');
matches = csvread('match.txt');

%find the homography matrix by using RANSAC
H2to1 = ransacH(matches,locs1,locs2,7000,10);
%stitch the images together without cliping any info
im3 = imageStitching_noClip(im1,im2,H2to1);
imshow(im3,[]);