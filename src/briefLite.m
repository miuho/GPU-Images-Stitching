function [ locs,desc ] = briefLite( im )
sig = 2;
k = sqrt(2);
levels = 1:6;
th_c = .03;
th_r = 12;


[DoGlocs, GaussianPyramid] = DoGdetector(im,sig,k,levels,th_c,th_r);


load('testPattern.mat');
[locs,desc] = computeBrief(im,GaussianPyramid,DoGlocs,k,levels,compareX,compareY);

end

