function [ DoGPyramid, DoGLevels ] = createDoGPyramid(gaussPyramid, levels )

%circular shift the pyramid around to align each pyramid with the pyramid
%one level down
gaussPyramidShift = circshift(gaussPyramid,1,3);
%subtract the different pyramids
DoGPyramid = gaussPyramid - gaussPyramidShift;
%throw out the first difference and first level as they are not meaningful
DoGPyramid = DoGPyramid(:,:,2:end);
DoGLevels = levels(2:end);

end

