

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>


///////////////////////////////////////////////////////////////////////////////
// CPU routine
///////////////////////////////////////////////////////////////////////////////

void scan_gold(float* odata, float* idata, const unsigned int len) 
{
  odata[0] = 0;
  for(int i=1; i<len; i++) odata[i] = idata[i-1] + odata[i-1];
}

///////////////////////////////////////////////////////////////////////////////
// GPU routine
///////////////////////////////////////////////////////////////////////////////

__global__ void scan(float *g_odata, float *g_idata, int n)
{
  extern __shared__ float temp[];  // allocated on invocation

  int thid = threadIdx.x;
  int offset = 1;

  temp[2*thid] = g_idata[2*thid]; // load input into shared memory
  temp[2*thid+1] = g_idata[2*thid+1];

  for (int d = n>>1; d > 0; d >>= 1)                    // build sum in place up the tree
  {
    __syncthreads();

    if (thid < d)
    {
      int ai = offset*(2*thid+1)-1;
      int bi = offset*(2*thid+2)-1;

      temp[bi] += temp[ai];
    }
    offset *= 2;
  }

  if (thid == 0) { temp[n - 1] = 0; } // clear the last element
  
  for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
  {
    offset >>= 1;
    __syncthreads();

    if (thid < d)
    {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;

        float t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
    }
  }

  __syncthreads();

  g_odata[2*thid] = temp[2*thid]; // write results to device memory
  g_odata[2*thid+1] = temp[2*thid+1];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_elements, mem_size, shared_mem_size;

  float *h_data, *reference;
  float *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_elements = 512;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 1000

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));

  // compute reference solution

  reference = (float*) malloc(mem_size);
  scan_gold( reference, h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, mem_size) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice));

  // execute the kernel

  shared_mem_size = sizeof(float) * (num_elements+1);
  scan<<<1,num_elements/2,shared_mem_size>>>(d_odata, d_idata, num_elements);
  getLastCudaError("scan kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, mem_size,
                              cudaMemcpyDeviceToHost) );

  // check results

  float err=0.0;
  for (int i = 0; i < num_elements; i++) {
    err += (h_data[i] - reference[i])*(h_data[i] - reference[i]);
//    printf(" %f %f \n",h_data[i], reference[i]);
  }
  printf("rms scan error  = %f\n",sqrt(err/num_elements));

  // cleanup memory

  free(h_data);
  free(reference);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
