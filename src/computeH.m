function [ H2to1 ] = computeH( p1,p2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
num_points = size(p1,2);
if(num_points < 4)
    disp('Less than 4 points, cannot compute homography');
end

%create x1 y1 altering in columns as they are needed on every other line
xy_col = p1(:);
%create 3 columns of them for use in the right side of A
xy_rep = repmat(xy_col,[1,3]);
%create a mask to multiply the u's and v's onto
xy_mask = [1,1,-1,0,0,0;0,0,0,1,1,-1];
xy_mask = repmat(xy_mask,[num_points,1]);
%append the mask with the xy columns
xy_mat = [xy_mask,xy_rep];

%create 2x2N mat with [Ui;Vi] occuring twice
uv_extend = [p2;p2];
uv_extend = reshape(uv_extend,2,2*num_points);
%create a ones mask to be in the spot where xy_mat doesn't have u's or v's
uv_mask = ones(1,2*num_points);
%combine the mask and the u's and v's
uv_mat = [-uv_extend;uv_mask;-uv_extend;uv_mask;uv_extend;uv_mask]';

%with the uv mat and the xy mat lined up, do an element wise multiply
A = uv_mat.*xy_mat;

%use SVD to solve for the homography matrix
[~,sin_val, r_sin_vec] = svd(A);
[~,vec_idx] = min(diag(sin_val));
H2to1 = r_sin_vec(:,vec_idx);
H2to1 = reshape(H2to1,3,3)';


end

