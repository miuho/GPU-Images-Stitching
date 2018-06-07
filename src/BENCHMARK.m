% MATLAB implementation for debugging the cuda version as well as
% benchmarking
% NOTE: Put in the same directory as all the other MATLAB functions. Two
% data files' directories also need to be specified.
clear; load testPattern.mat;

DEBUG = 0; % put 1 to enable debugging
TIMER = 10; % put a number to enable timer. The value is the number of iterations.
SINGLETHREAD = 1; % put 1 to enable single thread
IMAGE = {'incline_L.dat';'incline_R.dat'};
FILTER = 'filters.dat';
levels = 1:6;
th_contrast = .03;
th_r = 12;
ratio = 0.5;

LOCS = cell(size(IMAGE));
DESC = cell(size(IMAGE));

if SINGLETHREAD,
    maxNumCompThreads(1);
    fprintf('Thread Number = %d\n',maxNumCompThreads);
else
    maxNumCompThreads('automatic');
    fprintf('Thread Number = %d\n',maxNumCompThreads);
end

%% I/O
if TIMER, 
    TIMER_IO = zeros(length(IMAGE),TIMER);
    TIMER_GAUSSIAN_PYRAMID = zeros(length(IMAGE),TIMER);
    TIMER_DOG_PYRAMID = zeros(length(IMAGE),TIMER);
    TIMER_PRINCIPLE_CURVATURE = zeros(length(IMAGE),TIMER);
    TIMER_NMS = zeros(length(IMAGE),TIMER);
    TIMER_BRIEF = zeros(length(IMAGE),TIMER);
    TIMER_MATCH = zeros(1,TIMER);
    tic; 
end
for iter = 1: TIMER, % For timing purpose
for image_idx = 1:length(IMAGE),
    fprintf('%s:\n',IMAGE{image_idx});
    image = csvread(IMAGE{image_idx});

    if DEBUG, imageFFT = fft2(image); end

    filter_in = csvread(FILTER);

    filters = cell(1,6); % initilize holders
    if DEBUG, % intermediate steps in cuda for debugging
    filters_FFT = {};
    filters_shifted = {}; % padded & shifted filters
    filters_shifted_FFT = {};
    filters_shifted_FFT2 = {}; % alternative method by FT properties
    end
    i = 1; % filter_in row counter
    j = 1; % filters cell counter
    while i <= size(filter_in,1),
        if sum(filter_in(i,:)~=0) == 2, % hard code
            rows = filter_in(i,1);
            cols = filter_in(i,2);
            filters{j} = filter_in(i+1:i+rows,1:cols);
            if DEBUG, % intermediate steps in cuda for debugging
            filters_FFT{j} = fft2(filters{j},size(image,1),size(image,2));
            filter_padded = padarray(filters{j},size(image)-rows,'post');
            filter_padded_shifted = circshift(filter_padded,-floor(rows/2));
            filter_padded_shifted = circshift(filter_padded_shifted,-floor(rows/2),2);
            filters_shifted{j} = filter_padded_shifted;
            filters_shifted_FFT{j} = fft2(filter_padded_shifted);
            w1 = 2*pi/size(image,1)*(0:size(image,1)-1);
            w2 = 2*pi/size(image,2)*(0:size(image,2)-1);
            time_shift = floor(rows/2);
            fft_mask = (exp(1j*time_shift.*w1).') * exp(1j*time_shift.*w2);
            filters_shifted_FFT2{j} = filters_FFT{j} .* fft_mask;
            end
            i = i +rows+1;
            j = j + 1;
        end
    end
    if TIMER, TIMER_IO(image_idx,iter) = toc;  end
    %% Gaussian Pyramid (=> Filtering in Cuda version)
    if TIMER, tic; end
    GaussianPyramid = zeros([size(image),length(filters)]);
    if DEBUG, GaussianPyramid2 = cell(size(filters)); end
    DoGPyramid = zeros([size(image),length(filters)-1]);
    for i = 1:length(filters),
        GaussianPyramid(:,:,i) = imfilter(image,filters{i},'circular');
        if DEBUG, GaussianPyramid2{i} = ifft2(imageFFT.*filters_shifted_FFT{i});end
    end
    if TIMER, TIMER_GAUSSIAN_PYRAMID(image_idx,iter) = toc; tic; end
    for i = 1:length(filters),
        if i > 1, DoGPyramid(:,:,i-1) = GaussianPyramid(:,:,i) - GaussianPyramid(:,:,i-1); end
    end
    if TIMER, TIMER_DOG_PYRAMID(image_idx,iter) = toc; tic; end

    %% Compute curvature (=> gradient & curvature in cuda version)
    PrincipleCurvature = ...
        computePrincipleCurvature(DoGPyramid);
    if TIMER, TIMER_PRINCIPLE_CURVATURE(image_idx,iter) = toc; tic; end
    
    %% Non-local Maximal Suppression (=> nms in cuda version)
    locsDoG = getLocalExtrema(DoGPyramid,levels(2:end),PrincipleCurvature,...
    th_contrast,th_r);
    if TIMER, TIMER_NMS(image_idx,iter) = toc; tic; end
    
    %% Brief Feature extraction (=> brief feature in cuda version) 
    [locs,desc] = ...
        computeBrief(image,GaussianPyramid,locsDoG,0,levels,compareX,compareY);
    if TIMER, TIMER_BRIEF(image_idx,iter) =toc; end
    LOCS{image_idx} = locs;
    DESC{image_idx} = desc;
end
end

%% Comparison & match (=> compareA_to_B in cuda version)
for iter = 1:TIMER,
    if TIMER, tic; end
    matches = briefMatch(DESC{1}, DESC{2}, ratio);
    if TIMER, TIMER_MATCH(iter) = toc; end
end

% Compute average runtime
if TIMER,
    for i = 1:length(IMAGE),
        fprintf('Result of Image %d...\n',i);
        fprintf('Runtime Performance by Averaging %d runs:\n',TIMER);
        fprintf('I/O                :   %f seconds.\n', mean(TIMER_IO(i,:)));
        fprintf('Gaussian Pyramid   :   %f seconds.\n', mean(TIMER_GAUSSIAN_PYRAMID(i,:)));
        fprintf('DoGPyramid         :   %f seconds.\n', mean(TIMER_DOG_PYRAMID(i,:)));
        fprintf('Principle Curvature:   %f seconds.\n', mean(TIMER_PRINCIPLE_CURVATURE(i,:)));
        fprintf('NMS                :   %f seconds.\n', mean(TIMER_NMS(i,:)));
        fprintf('Brief Feature      :   %f seconds.\n', mean(TIMER_BRIEF(i,:)));
    end
    fprintf('Feature Match      :   %f seconds.\n', mean(TIMER_MATCH));
end