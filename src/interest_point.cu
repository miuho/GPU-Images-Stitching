/*
    interest_point.cu: Source file for cuda version of interest point matching
*/

#include "interest_point.h"


namespace cuda
{
  static cufftReal *imgData;
  static cufftReal *filtData;
  static cufftComplex *imgFFT;
  static cufftComplex *filtFFT;
  static cufftHandle forwardPlan;
  static cufftHandle inversePlan;

  static float *DoGPyramid;
  static float *Dx, *Dy, *Dxx, *Dxy, *Dyy, *PrincipleCurvature; // gradients & curvature
  static int *extrema_3D_indices, *extrema_count;


  //Starts a timer with a name for timing.  DOES NOT MAKE A COPY OF NAME
  struct timer startTimer(const char* name){
    struct timer timer;
    timer.name = name;
    cudaEventCreate(&timer.start);
    cudaEventCreate(&timer.stop);
    cudaEventRecord(timer.start);
    return timer;
  }

  void endAndPrintTime(struct timer timer){
      cudaEventRecord(timer.stop);
      cudaEventSynchronize(timer.stop);
      float milliseconds = 0;
      cudaEventElapsedTime(&milliseconds,timer.start,timer.stop);
    std::cout << "Timer " << timer.name << " took " << milliseconds << " ms to run.\n";
  }
  void debug_print_host(float *arr, int len) {
    for (int i = 0; i < len; i++)
      std::cout << arr[i] << ",";
    std::cout << "\n";
  }
  void debug_print_host_int(int *arr, int len) {
    for (int i = 0; i < len; i++)
      std::cout << arr[i] << ",";
    std::cout << "\n";
  }

  void debug_print_device(float *arr, int len) {
    float *arr2 = (float *)malloc(len * sizeof(float));
    cudaMemcpy(arr2, arr, len * sizeof(float), cudaMemcpyDeviceToHost);
    debug_print_host(arr2, len);
  }

  void debug_print_device_int(int *arr, int len) {
    int *arr2 = (int *)malloc(len * sizeof(int));
    cudaMemcpy(arr2, arr, len * sizeof(int), cudaMemcpyDeviceToHost);
    debug_print_host_int(arr2, len);
  }

  __global__ void
  element_wise_multiply_kernel(float *arr_1, float *arr_2, int len, float *arr_3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
      float a = arr_1[2*i];
      float b = arr_1[2*i + 1];
      float c = arr_2[2*i];
      float d = arr_2[2*i + 1];

      arr_3[2*i] = a * c - b * d;
      arr_3[2*i + 1] = a * d + b * c;
    }
  }

  __global__ void
  compute_curvature_kernel(float *dxx_arr, float *dxy_arr, float *dyy_arr,
                           int len, float *curv_arr, int numFilters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len) {
        for (int j = 0; j < numFilters-1; j++){
            int idx = j*len+i;
            float dxx = dxx_arr[idx];
            float dxy = dxy_arr[idx];
            float dyx = dxy;
            float dyy = dyy_arr[idx];
            float dxx_plus_dyy = dxx + dyy;
            curv_arr[idx] = (dxx_plus_dyy * dxx_plus_dyy) / (dxx * dyy - dxy * dyx);
        }
    }
  }

  __global__ void
  create_DoG_Pyramid_kernel(float *data_in, float *data_out,
                            int rows, int cols, int numFilters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols){ // within one image size
        int img_offset = rows * cols;
        for (int j = 0; j < numFilters-1; j++) {
            data_out[j*img_offset+i] =
                data_in[(j+1)*img_offset+i] - data_in[j*img_offset+i];
        }
    }
  }

  __global__ void
  gradient_kernel(float *data_in, float *dx_out, float *dy_out,
                  int rows, int cols, int numFilters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) {
        int img_offset = rows * cols;
        int col_num = i % cols;
        int row_num = i / cols;
        int idx;
        if ((col_num == 0) && (row_num == 0)){ // top left corner
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx+1] - data_in[idx];
                dy_out[idx] = data_in[idx+cols]-data_in[idx];
            }
        }
        else if ((col_num == cols-1) && (row_num == 0)){ // top right corner
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx] - data_in[idx-1];
                dy_out[idx] = data_in[idx+cols]-data_in[idx];
            }
        }
        else if ((col_num == 0) && (row_num == rows-1)){ // bottom left corner
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx+1] - data_in[idx];
                dy_out[idx] = data_in[idx]-data_in[idx-cols];
            }
        }
        else if ((col_num == 0) && (row_num == 0)){ // bottom right corner
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx] - data_in[idx-1];
                dy_out[idx] = data_in[idx]- data_in[idx-cols];
            }
        }
        else if (col_num == 0){ //  left edge
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx+1] - data_in[idx];
                dy_out[idx] = 0.5 * (data_in[idx+cols] - data_in[idx-cols]);
            }
        }
        else if (col_num == cols-1){ //  right edge
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = data_in[idx] - data_in[idx-1];
                dy_out[idx] = 0.5 * (data_in[idx+cols] - data_in[idx-cols]);
            }
        }
        else if (row_num == 0){ //  top edge
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = 0.5 * (data_in[idx+1] - data_in[idx-1]);
                dy_out[idx] = data_in[idx+cols]- data_in[idx];
            }
        }
        else if (row_num == rows-1){ //  bottom edge
            for (int j = 0; j < numFilters-1; j++){
                idx = j*img_offset+i; // starting index
                dx_out[idx] = 0.5 * (data_in[idx+1] - data_in[idx-1]);
                dy_out[idx] = data_in[idx]- data_in[idx-cols];
            }
        }
        else {
            for (int j = 0; j < numFilters-1; j++){ // non-edge
                idx = j*img_offset+i; // starting index
                dx_out[idx] = 0.5 * (data_in[idx+1] - data_in[idx-1]);
                dy_out[idx] = 0.5 * (data_in[idx+cols] - data_in[idx-cols]);
            }
        }

    }

  }



  __device__ float get_elem(float *arr, int width, int height, int i, int j) {
    return (i >= 0 && i < width &&  j >= 0 && j < height) ? arr[j*width+i] : 0;
  }

  __device__ void set_elem(char *arr, int width, int height, int i, int j, char elem) {
    if (i >= 0 && i < width &&  j >= 0 && j < height) { arr[j*width+i] = elem;}
  }

  __device__ void
  update_extrema_indices(int *extrema_3D_indices, int *extrema_count, int width, int height,
                         int x, int y, int z, char is_extrema) {
    if (!is_extrema)
      return;
    if ((x - 4) < 0 || (y - 4) < 0)
      return;
    if ((x + 4) >= width || (y + 4) >= height)
      return;
    int extrema_index = 3 * atomicAdd(extrema_count, 1);
    //printf("Extrema_index: %i, xyz = %i %i %i\n",extrema_index,x,y,z);
    extrema_3D_indices[extrema_index] = x;
    extrema_3D_indices[extrema_index + 1] = y;
    extrema_3D_indices[extrema_index + 2] = z;
  }

  __global__ void
  compute_local_extrema_kernel(float *dog_arr, float *curve_arr,
                               int width, int height, int dist_to_center,
                               float contrast_th,float curve_th, int numFilters,
                               int *extrema_3D_indices, int *extrema_count) {
    // each cuda block handles one block (32 x 32 pixel) of the image
    // and each block overlaps with adjacent blocks with dist_to_center
    int xidx = blockIdx.x * blockDim.x + threadIdx.x;
    int yidx = blockIdx.y * blockDim.y + threadIdx.y;
    int pad_xidx = xidx - (2*blockIdx.x+1)*dist_to_center;
    int pad_yidx = yidx - (2*blockIdx.y+1)*dist_to_center;
    __shared__ float A[BLOCK_SIZE][BLOCK_SIZE];
    //int pad_idx = pad_yidx * width + pad_xidx; // linear index for each block

    for (int f = 0; f < numFilters; f++) { // loop through each image
        float* dog_cur = dog_arr + width*height*f; // current image
        A[threadIdx.y][threadIdx.x] = get_elem(dog_cur, width,height, pad_xidx,pad_yidx);
        __syncthreads();
        if (threadIdx.x >= dist_to_center && threadIdx.x < blockDim.x-dist_to_center &&
            threadIdx.y >= dist_to_center && threadIdx.y < blockDim.y-dist_to_center) {
                    float* dog_prev = (f==0) ? NULL : (dog_cur - (width*height));
                    float* dog_next = (f==numFilters-1) ? NULL : (dog_cur + (width*height));
                    float center = A[threadIdx.y][threadIdx.x];
                    char flag = 1; // 1 if max, 0 otherwise
                    //*
                    if ((dog_prev != NULL && dog_prev[pad_yidx*width+pad_xidx] > center) ||
                        (dog_next != NULL && dog_next[pad_yidx*width+pad_xidx] > center)){
                        // check across images
                        flag = 0;
                        goto eol;
                    }
                    //*/
                    float cur;
                    for (int i = threadIdx.y-dist_to_center; i <= threadIdx.y+dist_to_center; i++) {
                      for (int j = threadIdx.x-dist_to_center; j <= threadIdx.x+dist_to_center; j++) {
                          if (threadIdx.y != i || threadIdx.x != j){ // ignore self
                              cur = A[i][j];
                              if (cur > center) {
                                  flag = 0;
                                  goto eol;
                              }
                          }
                      }
                    }
eol:        update_extrema_indices(extrema_3D_indices, extrema_count, width, height, pad_xidx, pad_yidx, f,
                         flag && (center > contrast_th) &&
                             (curve_arr[pad_yidx * width + pad_xidx] < curve_th));
            }
            __syncthreads();
        }
  }

  __device__ int
  bit_diff_count(uint32_t a, uint32_t b) {
    int diffBits = a ^ b;
    return __popc(diffBits);
  }

  __device__ int
  compute_diff(brief_t *A, brief_t *B) {
    int count = bit_diff_count(A->a, B->a);
    count += bit_diff_count(A->b, B->b);
    count += bit_diff_count(A->c, B->c);
    count += bit_diff_count(A->d, B->d);
    count += bit_diff_count(A->e, B->e);
    count += bit_diff_count(A->f, B->f);
    count += bit_diff_count(A->g, B->g);
    count += bit_diff_count(A->h, B->h);
    return count;
  }

  __global__ void
  compare_A_to_B(brief_t *arr_A, int len_A, brief_t *arr_B, int len_B, int *arr_C, float threshold) {
    // arr_A should have same length as arr_C
    __shared__ brief_t arr_B_s[1024];
    __shared__ int min_diff_indices1[32 * 32]; // 32 warps in each block
    __shared__ int min_diff_indices2[32 * 32]; // 32 warps in each block
        int global_min_diff_value1 = INT_MAX;
        int global_min_diff_value2 = INT_MAX;
        int global_min_diff_index1 = -1;
        int global_min_diff_index2 = -1;
  while (len_B > 0) {
    // load arr_B to shared memory so that each block has a copy of arr_B
    if (threadIdx.x < len_B) {
      arr_B_s[threadIdx.x].a = arr_B[threadIdx.x].a;
      arr_B_s[threadIdx.x].b = arr_B[threadIdx.x].b;
      arr_B_s[threadIdx.x].c = arr_B[threadIdx.x].c;
      arr_B_s[threadIdx.x].d = arr_B[threadIdx.x].d;
      arr_B_s[threadIdx.x].e = arr_B[threadIdx.x].e;
      arr_B_s[threadIdx.x].f = arr_B[threadIdx.x].f;
      arr_B_s[threadIdx.x].g = arr_B[threadIdx.x].g;
      arr_B_s[threadIdx.x].h = arr_B[threadIdx.x].h;
    }
    __syncthreads();

    // each warp (32 threads) handles one element in arr_A
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < (len_A * 32)) {
      int A_i = thread_id / 32; // this is the id of element in arr_A
      int warp_id = A_i % 32; // this is the warp id within a block
      int part_id = thread_id % 32; // this is the id of threads in same warp
      brief_t A;
      A.a = arr_A[A_i].a;
      A.b = arr_A[A_i].b;
      A.c = arr_A[A_i].c;
      A.d = arr_A[A_i].d;
      A.e = arr_A[A_i].e;
      A.f = arr_A[A_i].f;
      A.g = arr_A[A_i].g;
      A.h = arr_A[A_i].h;
      int min_diff_value1 = INT_MAX;
      int min_diff_value2 = INT_MAX;
      int min_diff_index1 = -1;
      int min_diff_index2 = -1;

      // each thread in a warp handles one part of arr_B for the same element in arr_A
      while (part_id < len_B && part_id < 1024) {
        int cur_diff = compute_diff(&A, &arr_B_s[part_id]);
        // cur_diff is smaller than both 1st and 2nd
        if (cur_diff <= min_diff_value1 && cur_diff <= min_diff_value2) {
          min_diff_value2 = min_diff_value1;
          min_diff_value1 = cur_diff;
          min_diff_index2 = min_diff_index1;
          min_diff_index1 = part_id;
        }
        // cur_diff is smaller than 2nd only
        else if (cur_diff >= min_diff_value1 && cur_diff <= min_diff_value2) {
          min_diff_value2 = cur_diff;
          min_diff_index2 = part_id;
        }
        else {}
        part_id += 32;
      }

      min_diff_indices1[threadIdx.x] = min_diff_index1;
      min_diff_indices2[threadIdx.x] = min_diff_index2;
      __syncthreads();
      // each warp here collapse 32 minimum values
      if (thread_id % 32 == 0) {
       for (int i = 0; i < 64; i++) {
          int index;
          if (i < 32)
            index = min_diff_indices1[warp_id * 32 + i];
          else
            index = min_diff_indices2[warp_id * 32 + i - 32];
          if (index >= 0) {
            int cur_diff = compute_diff(&A, &arr_B_s[index]);
            // cur_diff is smaller than both 1st and 2nd
            if (cur_diff <= global_min_diff_value1 && cur_diff <= global_min_diff_value2) {
              global_min_diff_value2 = global_min_diff_value1;
              global_min_diff_value1 = cur_diff;
              global_min_diff_index2 = global_min_diff_index1;
              global_min_diff_index1 = index;
            }
            // cur_diff is smaller than 2nd only
            else if (cur_diff >= global_min_diff_value1 && cur_diff <= global_min_diff_value2) {
              global_min_diff_value2 = cur_diff;
              global_min_diff_index2 = index;
            }
            else {}
          }
        }
        float ratio = ((float)global_min_diff_value1)/((float)global_min_diff_value2);
        if (ratio < threshold)
          arr_C[A_i] = global_min_diff_index1;
        else
          arr_C[A_i] = -1;
      }
    }

    arr_B += 1024;
    len_B -= 1024;
  }
  }

  __global__ void
  get_brief_feature_vec(float *dog_arr, int width, int height, int num_filt,
                        int *locs, brief_t *features){
    __shared__ float patch[9][9];
    __shared__ uint32_t comps[NUM_BRIEF_COMPS];

    if(threadIdx.x < 81){
      int centerX = locs[3*blockIdx.x];
      int centerY = locs[3*blockIdx.x + 1];
      int centerZ = locs[3*blockIdx.x + 2];

      int x = threadIdx.x%9;
      int y = threadIdx.x/9;

      int x_act = centerX + x - 4;
      int y_act = centerY + y - 4;

      patch[y][x] = dog_arr[centerZ*width*height + y_act*width + x_act];
    }
    __syncthreads();
    comps[threadIdx.x] = ((patch[compare_locs_device[threadIdx.x].y1]
                          [compare_locs_device[threadIdx.x].x1] <
                          patch[compare_locs_device[threadIdx.x].y2]
                          [compare_locs_device[threadIdx.x].x2]) <<
                          ((255-threadIdx.x)%32));
    __syncthreads();
    for(int mod = 2; mod <= 32; mod*=2){
      if(!(threadIdx.x % mod))
        comps[threadIdx.x] |= comps[threadIdx.x + mod/2];

    }
    __syncthreads();
    if(threadIdx.x < 8)
      *((uint32_t*)(features + blockIdx.x) +threadIdx.x) = comps[32*threadIdx.x];

  }


  int
  interestPointMatch(brief_t *features, float *img, float* filt, int numRows, int numCols,
                     int numFilters, char img_name[]){

    
    
    int i;
    cudaMemcpy(filtData, filt,sizeof(cufftReal)*numCols*numRows*numFilters,
               cudaMemcpyHostToDevice);


    struct timer TIMER_GaussianPyramid = startTimer("Gaussian Pyramid");
    cudaMemcpy(imgData, img,sizeof(cufftReal)*numCols*numRows,
                          cudaMemcpyHostToDevice);
    //FFT of current image
    if (cufftExecR2C(forwardPlan, imgData, imgFFT) != CUFFT_SUCCESS){
      fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
      return -1;
    }


    //Filter current Image with set of filters
    int len = numRows * (numCols/2+1);
    //int size = len * 2 * sizeof(float);

    dim3 dimBlock(1024);
    dim3 dimGrid((len + 1023)/1024);
    for(i = numFilters-1; i >= 0; i--){
      float *curFiltFFT = (float *)(filtFFT + len*i);
      float *curImgFFT = (float *)(imgFFT + len*i);
      element_wise_multiply_kernel<<<dimGrid, dimBlock>>>((float *)imgFFT,
                                                          curFiltFFT, len,
                                                          curImgFFT);
    }

    //Take the inverse FFT
    for(i = 0; i < numFilters; i++){
      cufftComplex* curImgFFT = imgFFT + len*i;
      cufftReal* curImgData = imgData + numRows*numCols*i;
      if (cufftExecC2R(inversePlan, curImgFFT, curImgData) != CUFFT_SUCCESS){
        fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
        return -1;
      }
    }
    endAndPrintTime(TIMER_GaussianPyramid);
    struct timer TIMER_DoGPyramid = startTimer("DoG Pyramid");
    // Compute first order difference of pyramids. CHECK
    dim3 dimBlock_dog(1024);
    dim3 dimGrid_dog((numRows*numCols+1023)/1024);
    create_DoG_Pyramid_kernel<<<dimGrid_dog,dimBlock_dog>>>((float *)imgData, DoGPyramid,
                              numRows, numCols, numFilters);
    endAndPrintTime(TIMER_DoGPyramid);

    struct timer TIMER_PrincipleCurvature = startTimer("Principle Curvature");
    // Compute gradients
    dim3 dimBlock_grad(1024);
    dim3 dimGrid_grad((numRows*numCols+1023)/1024);
    gradient_kernel<<<dimGrid_grad,dimBlock_grad>>>(DoGPyramid, Dx, Dy, numRows, numCols,numFilters);
    gradient_kernel<<<dimGrid_grad,dimBlock_grad>>>(Dx, Dxx, Dxy, numRows, numCols,numFilters);
    gradient_kernel<<<dimGrid_grad,dimBlock_grad>>>(Dy, Dxy, Dyy, numRows, numCols,numFilters);
    compute_curvature_kernel<<<dimGrid_grad,dimBlock_grad>>>
        (Dxx, Dxy, Dyy,numRows*numCols, PrincipleCurvature,numFilters);
    endAndPrintTime(TIMER_PrincipleCurvature);

    struct timer TIMER_nms = startTimer("Non-local Maximal Suppression");
    dim3 dimBlock_nms(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid_nms(((numCols+BLOCK_SIZE-1)/BLOCK_SIZE),((numRows+BLOCK_SIZE-1)/BLOCK_SIZE));
    compute_local_extrema_kernel<<<dimGrid_nms,dimBlock_nms>>>
        (DoGPyramid, PrincipleCurvature, numCols, numRows, NMS_DIST_TO_CENTER,
        CONTRAST_THRESHOLD*numRows*numCols,CURVATURE_THRESHOLD,numFilters,extrema_3D_indices,extrema_count);
    endAndPrintTime(TIMER_nms);

    int extrema_count_host;
    cudaMemcpy(&extrema_count_host, extrema_count,sizeof(int),cudaMemcpyDeviceToHost);
    if (DEBUG) printf("Total Count: %d\n",extrema_count_host);
    
    // Feature extraction
    struct timer TIMER_brief = startTimer("Brief Feature Extraction");
    get_brief_feature_vec<<<extrema_count_host,256>>>
            (imgData, numCols, numRows, numFilters,extrema_3D_indices, features);
    endAndPrintTime(TIMER_brief);

    if (DEBUG){
      printf("Features:\n");
      debug_print_device_int((int*)features,24);
    }
    if(writeOutput){
      /*------------- Output to Files --------------*/
      int *extrema_3D_indices_host;
      extrema_3D_indices_host = (int*)malloc(extrema_count_host*3*sizeof(int));
      cudaMemcpy(extrema_3D_indices_host,extrema_3D_indices,extrema_count_host*3*sizeof(int),cudaMemcpyDeviceToHost);

      std::ofstream output;
      output.open(img_name);
      for (int i = 0; i < extrema_count_host*3; i+=3){
        output << (1+extrema_3D_indices_host[i]) << "," <<
          (1+extrema_3D_indices_host[i+1]) << "," <<
          (1+extrema_3D_indices_host[i+2]) << "\n";
      }
      output.close();
      free(extrema_3D_indices_host);
    }
    return extrema_count_host;
  }


  void interestPointInitialize(float *img1, float *img2, float* filts, int numRows, int numCols, int numFilters, int writeToFile){
          brief_t *features1;
          brief_t *features2;
          int *match_indices, *match_indices_host;
          

          writeOutput = writeToFile;
          cudaMalloc((void**)&features1, sizeof(brief_t)*numCols*numRows);
          cudaMalloc((void**)&features2, sizeof(brief_t)*numCols*numRows);
          cudaMalloc((void**)&match_indices, sizeof(int)*numCols*numRows);
          cudaMemset((void*)match_indices, 0, sizeof(int)*numCols*numRows);
          cudaMalloc((void**)&extrema_3D_indices, sizeof(int)*numCols*numRows*3);
          cudaMemset((void*)extrema_3D_indices, 0, sizeof(int)*numCols*numRows*3);
          cudaMalloc((void**)&extrema_count, sizeof(int));
          cudaMemset((void*)extrema_count, 0, sizeof(int));
          cudaMalloc((void**)&imgData, sizeof(cufftReal)*numCols*numRows*numFilters);
          cudaMalloc((void**)&filtData, sizeof(cufftReal)*numCols*numRows*numFilters);
          cudaMalloc((void**)&imgFFT, sizeof(cufftComplex)*numRows*(numCols/2+1)*numFilters);
          cudaMalloc((void**)&filtFFT, sizeof(cufftComplex)*numRows*(numCols/2+1)*numFilters);
          cudaMalloc((void**)&DoGPyramid, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&Dx, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&Dy, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&Dxx, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&Dxy, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&Dyy, sizeof(float)*numCols*numRows*(numFilters-1));
          cudaMalloc((void**)&PrincipleCurvature, sizeof(float)*numCols*numRows*(numFilters-1));


          // Calculate Filter FFTs, only needs to be done once at the begining.
          // TODO: USE CONSTANT (or maybe texture) MEMORY TO HOLD THE FILTERS AFTER
          int n[NUM_DIMS] = {numRows,numCols};
          int i;
          //Plan the forward an inverse FFTs
          if (cufftPlan2d(&forwardPlan, n[0], n[1],CUFFT_R2C) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            return;
          }
          if (cufftPlan2d(&inversePlan, n[0], n[1],CUFFT_C2R) != CUFFT_SUCCESS){
            fprintf(stderr, "CUFFT Error: Unable to create plan\n");
            return;
          }
          // COMPUTING THEM ONCE
          for(i = 0; i < numFilters; i++){
            cufftReal* curFilt = filtData + numCols*numRows*i;
            cufftComplex* curFiltFFT = filtFFT + numRows*(numCols/2+1)*i;
            if (cufftExecR2C(forwardPlan, curFilt, curFiltFFT) !=
                CUFFT_SUCCESS){
              fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
              return;
            }
          }
          /*END SINGLE EXECUTION SECTION*/


          int features1_len = interestPointMatch(features1,img1,filts,numRows,numCols,numFilters,"image1.txt");
          cudaMemset((void*)match_indices, 0, sizeof(int)*numCols*numRows);
          cudaMemset((void*)extrema_3D_indices, 0, sizeof(int)*numCols*numRows*3);
          cudaMemset((void*)extrema_count, 0, sizeof(int));
          int features2_len = interestPointMatch(features2,img2,filts,numRows,numCols,numFilters,"image2.txt");
          if (DEBUG){
            std::cout << features1_len << "\n";
            std::cout << features2_len << "\n";
          }
          struct timer TIMER_match = startTimer("Feature Comparison & Match");
          compare_A_to_B<<<(features1_len + BLOCK_SIZE - 1)/BLOCK_SIZE, BLOCK_SIZE*BLOCK_SIZE>>>
              (features1, features1_len, features2, features2_len, match_indices, RATIO_THRESHOLD);
          //cudaDeviceSynchronize();
          endAndPrintTime(TIMER_match);
          if (DEBUG) std::cout << cudaGetErrorString(cudaPeekAtLastError()) << "\n";

          if (DEBUG) debug_print_device_int(match_indices,features1_len);

          if(writeOutput){

            /*------------- Output to Files --------------*/
            //cudaDeviceSynchronize();
            match_indices_host = (int*)malloc(features1_len*sizeof(int));
            cudaMemcpy(match_indices_host,match_indices,features1_len*sizeof(int),cudaMemcpyDeviceToHost);
            std::ofstream output;
            output.open("match.txt");
            for (int i = 0; i < features1_len; i++){
              if (match_indices_host[i] != -1) {
                output << (i+1) << "," << (match_indices_host[i]+1) << "\n";
              }
            }
            output.close();
          }
          // Free allocated space
          cudaFree(features1);
          cudaFree(features2);
          cudaFree(match_indices);
          cudaFree(imgData);
          cudaFree(filtData);
          cudaFree(imgFFT);
          cudaFree(filtFFT);
          cudaFree(DoGPyramid);
          cudaFree(Dx);
          cudaFree(Dy);
          cudaFree(Dxx);
          cudaFree(Dxy);
          cudaFree(Dyy);
          cudaFree(PrincipleCurvature);
          cudaFree(extrema_3D_indices);
          cudaFree(extrema_count);
  }


} // namespace cuda
