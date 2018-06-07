function generatePanorama()
im1 = csvread('incline_L.dat');
im2 = csvread('incline_R.dat');
%get the location and descriptor for each image
[locs1,desc1] = briefLite(im1);
[locs2,desc2] = briefLite(im2);
%match the descriptors but require a high ratio between best and second
%best
matches = briefMatch(desc1,desc2,.5);
figure; plotMatches(im1, im2, matches, locs1, locs2);
%find the homography matrix by using RANSAC
H2to1 = ransacH(matches,locs1,locs2,5000,10);
%stitch the images together without cliping any info
im3 = imageStitching_noClip(im1,im2,H2to1);
figure; imshow(im3,[]);
end

