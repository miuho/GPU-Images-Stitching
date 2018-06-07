function [bestH] = ransacH(matches, locs1, locs2, nIter, tol)
if(~exist('tol'))
    tol = 10;
end
if(~exist('nIter'))
    nIter = 1000;
end

%set the tolerance as a percentage of area the matches occur over
%overlap_size = max(locs1(matches(1,:),1)) - min(locs1(matches(1,:),1));
tol_dist = tol;%overlap_size*tol;


%set variables
num_points = size(matches,1);
num_ransac_points = 6;
p_extender = ones(1,num_points);
num_inliers = 0;
bestH = zeros(3);
%create matched pairs
p1 = locs1(matches(:,1),1:2)';
p2 = locs2(matches(:,2),1:2)';
for i = 1:nIter
    %choose a random sampling of matches and compute the H matrix
    rand_idxs = randperm(num_points, num_ransac_points);
    rand_p1 = p1(:,rand_idxs);
    rand_p2 = p2(:,rand_idxs);
    curH = computeH(rand_p1,rand_p2);
    
    %map all points in 2->1 based on H
    p2_ext = [p2;p_extender];
    p1_guess = curH*p2_ext;
    lambda_mat = p1_guess(3,:);
    lambda_mat = repmat(lambda_mat,[3,1]);
    p1_guess = p1_guess./lambda_mat;
    p1_guess = p1_guess(1:2,:);
    
    %calculate the distance of the guess from the real point
    D = (p1_guess - p1).^2;
    D = sqrt(D(1,:) +  D(2,:));
    
    %count the total that match well and replace H if it's better than
    %previous tries
    cur_num_inliers = sum(D < tol_dist);
    if(cur_num_inliers > num_inliers)
        num_inliers = cur_num_inliers;
        bestH = curH;
    end
end
disp(num_inliers)
end

