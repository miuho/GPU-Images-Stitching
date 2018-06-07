% Parameters
OUTFILE = 'filters.dat';
if exist(OUTFILE,'file') == 2, delete(OUTFILE); end
sigma0 = 1;
k = sqrt(2);
levels = 1:6;

h = cell(length(levels)*2,1);
for i = 1:length(levels)
    sigma_ = sigma0*k^levels(i);
    hi = fspecial('gaussian',floor(3*sigma_*2)+1,sigma_);
    h{2*i-1} = size(hi);
    h{2*i} = hi;
    dlmwrite(OUTFILE,h{2*i-1},'-append');
    dlmwrite(OUTFILE,h{2*i},'-append');
end
fprintf('Filters created.\n');
