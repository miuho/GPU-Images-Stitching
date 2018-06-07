% Read CUDA output and plot images and matches

clear;
%%{
im1 = csvread('incline_L.dat');
im2 = csvread('incline_R.dat');
locs1 = csvread('image1.txt');
locs2 = csvread('image2.txt');
matches = csvread('match.txt');

width = size(im1,2) + size(im2,2);
height = max(size(im1,1), size(im2,1));
nchannel = size(im1,3); 
img = zeros(height, width,nchannel);
%     size(img)
img(1:size(im1,1),1:size(im1,2),:) = im1;
img(1:size(im2,1),size(im1,2)+1:size(im1,2) + size(im2,2),:) = im2; 
imshow(img,[]);

axis equal;
hold on;
for i = 1:length(matches)
    p1 = locs1(matches(i,1),:);
    p2 = locs2(matches(i,2),:);
    hold on
    line([p1(1) size(im1,2)+p2(1)],[p1(2),p2(2)], 'Color','r','LineWidth',1); 
end
hold off; 