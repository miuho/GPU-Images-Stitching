function im2txt( path_input, path_output, normalize, scale )
%IM2MAT: Read a (possibly compressed) image file and output a plain text
%file containing a floating point matrix representing the image. 
% path_input - absolute path to the input file
% normalize  - put 1 to normalize dynamic range to [0,1], else put 0
% path_output= absolute path to the output file

img = double(imread(path_input));
if size(img,3) > 1, img = mean(img,3); end
if normalize, img = img / 255; end
if length(scale) == 2, img = imresize(img,scale); end
%dlmwrite(path_output,size(img),'-append');
dlmwrite(path_output,img);
fprintf('Image data for %s created.\n',path_input);
end

