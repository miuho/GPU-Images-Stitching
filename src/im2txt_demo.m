% convert image to text file
im2txt('incline_L.png','incline_L.dat',1,[512,1024]);
im2txt('incline_R.png','incline_R.dat',1,[512,1024]);
% read matrix from text and display as image
img1 = csvread('inclineL.dat');
img2 = csvread('inclineR.dat');
figure; imshow(img1,[]);
figure; imshow(img2,[]);