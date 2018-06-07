# GPU-Images-Stitching

Implementation of interest point detection for video object tracking on MATLAB and parallelized on GPU.

Lucas Crandall (lcrandal), HingOn Miu (hmiu), Raymond Xia (yangyanx)
Carnegie Mellon University


################################################################################
Running our MATLAB implementation:
* Creating input data files:
    >> create_filters
    >> im2txt('incline_L.png','incline_L.dat',1,[512,1024]);
    >> im2txt('incline_R.png','incline_R.dat',1,[512,1024]);
  The above lines will generate ‘filters.dat’, ‘incline_L.dat’, and ‘incline_R.dat’.
  Note that all three files have already been generated for your convenience.

* For baseline benchmarking:
    >> BENCHMARK
  This script will print out the result to the MATLAB console. This is the
  baseline performance of our project. You should see something like this:
    Result of Image 1…
Runtime Performance by Averaging 10 runs:
I/O                :   1.325522 seconds.
Gaussian Pyramid   :   0.432463 seconds.
DoGPyramid         :   0.040190 seconds.
Principle Curvature:   0.359634 seconds.
NMS                :   0.426080 seconds.
Brief Feature      :   2.095197 seconds.
Result of Image 2...
Runtime Performance by Averaging 10 runs:
I/O                :   2.366984 seconds.
Gaussian Pyramid   :   0.398385 seconds.
DoGPyramid         :   0.041317 seconds.
Principle Curvature:   0.326046 seconds.
NMS                :   0.394707 seconds.
Brief Feature      :   1.179285 seconds.
Feature Match      :   2.421474 seconds.

* Generate matched images and stitched images (panorama) in MATLAB:
    >> generatePanorama
  You can use this result to compare with the result of the CUDA version.
  If the result doesn’t appear to be good, re-run the program until it looks good.

################################################################################
Running our CUDA implementation:
NOTE: ‘filters.dat’, ‘incline_L.dat’, and ‘incline_R.dat’ have to be present.

$ make
$ ./interest_point -f ./filters.dat -n 6 -i1 incline_L.dat -i2 incline_R.dat -r 512 -c 1024 -o

The above two lines are the standard commands in the current setting. The general command to run the program is:
$ ./interest_point -f /PATH/TO/FILTERS -n NUM_FILTERS -i1 /PATH/TO/IMAGE1 -i2 /PATH/TO/IMAGE2 -r NUM_IMAGE_ROWS -c NUM_IMAGE_COLS [-o]

All arguments except -o are required, the -o option outputs the interest points
for each image to ‘image1.txt’ and ‘image2.txt’, while the index of matching points
are output to ‘match.txt’. You will need these three files to visualize the result in MATLAB.
Move the above three files to the same directory that you are running MATLAB from,
or use the three files we put in the directory.

The filters provided are in the file ‘filters.dat’ and the two example images
are in ‘incline_L.dat’ and ‘incline_R.dat’.  For the provided images and filters,
use -n 6 -c 1024 -r 512.

################################################################################
Visualizing CUDA Result in MATLAB:
* Generate matched images:
    >> display_match
* Generate Panorama:
    >> panorama

In both cases, you should get similar results as in the report.
