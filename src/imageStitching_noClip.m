function [panoImg] = imageStitching_noClip(img1, img2, H2to1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

img1_size = size(img1);
img2_size = size(img2);

%get the cornor locations of each image
cornors_1 = [1,1,img1_size(2),img1_size(2);1,img1_size(1),1,img1_size(1)];
cornors_2 = [1,1,img2_size(2),img2_size(2);1,img2_size(1),1,img2_size(1)];

%warp and round the cornors of image 2 into image 1
cornors_2 = [cornors_2;1,1,1,1];
cornors_2 = H2to1*cornors_2;
lambda_mat = repmat(cornors_2(3,:),[3,1]);
cornors_2 = cornors_2./lambda_mat;
cornors_2 = round(cornors_2(1:2,:));

%finsing the min and max of all the cornors will tell us where the images
%end
left_and_top = min([cornors_1, cornors_2],[],2); 
right_and_bot = max([cornors_1, cornors_2],[],2);

%total size is a differnce between the edges as they are not aligned to the
%origin
out_size = right_and_bot - left_and_top;
out_size = fliplr(out_size');
%find the location of the topleft corner relative to the origin, need to
%shift by that amount
trans_amt = [1;1] - left_and_top;

%create a translation matrix with the amount found
M = [[1,0;0,1;0,0],[trans_amt;1]];
%warp both images, outsize is now the same for both
warp_im1 = warpH(img1, M, out_size);
warp_im2 = warpH(img2, M*H2to1,out_size);

%conver both images to floats to take the max and convert back
warp_im1 = im2double(warp_im1);
warp_im2 = im2double(warp_im2);
panoImg = max(warp_im1,warp_im2);
panoImg = uint8(floor(256*panoImg));

end

