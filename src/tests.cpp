#include <iostream>
#include <string.h>
#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/TestResult.h>
#include <cppunit/TestResultCollector.h>
#include <cppunit/XmlOutputter.h>
#include "interest_point.h"
#include <fstream>
#include <stdlib.h>
#include <sstream>
#include <string>

#define NUM_REPS 30

int main(int argc, char **argv){
  using namespace std;
  int i = 0;
  int j = 0;
  int k = 0;
  int numCols = 0;
  int numRows = 0;
  char *img1_path;
  char *img2_path;
  char *fil_path;
  float* img1;
  float* img2;
  float* filts;
  int numFilters;
  int writeOutput = 0;
  for(i = 1; i < argc; i++){
    if(argv[i][0] == '-'){
      switch(argv[i][1]){
        case 'r':
          numRows = atoi(argv[++i]);
          break;
        case 'c':
          numCols = atoi(argv[++i]);
          break;
        case 'i': // first image
            switch(argv[i][2]){
                case '1':
                  img1_path = argv[++i];
                  break;
                case '2':
                  img2_path = argv[++i];
                  break;
            }
          break;
        case 'f':
          fil_path = argv[++i];
          break;
        case 'n':
          numFilters = atoi(argv[++i]);
          break;
        case 'o':
          writeOutput = 1;
        default:
          break;
      }
    }
  }

  img1 = (float*)malloc(numRows*numCols*sizeof(float));
  img2 = (float*)malloc(numRows*numCols*sizeof(float));
  //filts = (float*)malloc(numRows*numCols*sizeof(float)*numFilters);
  filts = (float*)calloc(numRows*numCols*numFilters,sizeof(float));

  ifstream in(img1_path);
  string line, field;
  i = 0;
  j = 0;
  for(i=0;getline(in,line);i++){
    stringstream ss(line);
    for(j = 0;getline(ss,field,',');j++){
      img1[i*numCols + j] = atof(field.c_str());
    }
  }

  ifstream in2(img2_path);
  i = 0;
  j = 0;
  for(i=0;getline(in2,line);i++){
    stringstream ss(line);
    for(j = 0;getline(ss,field,',');j++){
      img2[i*numCols + j] = atof(field.c_str());
    }
  }

  ifstream in1_fil(fil_path);
  int filtX;
  int filtY;
  float *curFilt;
  for(i = 0;i < numFilters;i++){
    curFilt = filts + numCols*numRows*i;
    getline(in1_fil,line);
    stringstream ss(line);
    getline(ss,field,',');
    filtX = atof(field.c_str());
    getline(ss,field,',');
    filtY = atof(field.c_str());

    int xShift = filtX/2;
    int yShift = filtY/2;

    int row_start = numRows - xShift; // starting index in padded array
    int col_start = numCols - yShift;
    for (j = 0; j < filtX; j++) { // each row of filter
        getline(in1_fil,line);
        stringstream ss(line);
        for (k = 0; getline(ss,field,','); k++) {
            curFilt[((row_start+j)%numRows)*numCols +
                (col_start+k)%numCols] = atof(field.c_str());
        }

    }
  }




  cuda::interestPointInitialize(img1, img2, filts, numRows, numCols,
                                numFilters, writeOutput);
}
