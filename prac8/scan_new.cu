#include <string.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <chrono>

#include <omp.h>
#include <mpi.h>
#include <helper_cuda.h>

using namespace std;

/*
* CPU Scan
* 
* scan_output: output scan array
* data: input array
* num_elements: size of input array
*/
void scan_cpu(float* scan_output, 
			float* data, 
			const unsigned int num_elements) 

{		
	scan_output[0] = data[0];
	for (int i = 1; i < num_elements; i++)
	{
		scan_output[i] = data[i-1] + scan_output[i-1];
	}
}

/*
* GPU Scan
* 
* scan_output: output scan array
* data: input array
*/
__global__ void scan_gpu(float *scan_output, 
						float *data)
{
	// Dynamically allocated shared memory for scan kernels
	extern __shared__ float tree_sum[];

	float temp_sum;
	int   tid = threadIdx.x;

	// read input into shared memory
	temp_sum     = data[tid];
	tree_sum[tid] = temp_sum;

	// scan up the tree
	for (int d = 1; d < blockDim.x; d = 2*d) 
	{
		__syncthreads();

		if (tid-d >= 0) 
		{
			temp_sum = temp_sum + tree_sum[tid-d];
		}

		__syncthreads();

		tree_sum[tid] = temp_sum;
	}

	// write results to global memory
	__syncthreads();
	if (tid == 0) 
	{
		temp_sum = data[0];
	}
	else 
	{
		temp_sum = tree_sum[tid];
	}
	
	scan_output[tid] = temp_sum;
}

/*
* Check the GPU scan solution against the CPU scan solution
* Prints out the root mean squared difference
*
* scan_output: output scan array
* data: input array
* num_elements: size of input array
*/
void check_solution(float* scan_output, 
					float* data, 
					const int num_elements)
{
	float reference[num_elements];

	// compute reference solution
	scan_cpu(reference, data, num_elements);

	float err = 0.0;
	for (int i = 0; i < num_elements; i++) 
	{
		err += pow(scan_output[i] - reference[i], 2);
		// printf(" %f %f \n", data[i], reference[i]);
	}
	printf("rms scan error = %f\n", sqrt(err/num_elements));
}

/*
* Initialize the numbers for an array
* The initialization can be either random or a constant 2
* 
* data: generated array
* num_elements: size of generated array
* rank: MPI rank
* random: whether to use random or constant number for the generated array
*/
void create_data(float* data, 
				const int num_elements, 
				const int rank, 
				const bool random)
{
	srand(0);

	// Initialize only on the first processor
	if (rank == 0)
	{
		if (random)
		{
			#pragma omp parallel for
			for (int i = 0; i < num_elements; i++) 
			{
				data[i] = floorf(1000*(rand()/(float)RAND_MAX));
			}
		}
		else
		{
			#pragma omp parallel for
			for (int i = 0; i < num_elements; i++) 
			{
				data[i] = 2;
			}
		}
	}

	// Broadcast the data to all other processors to ensure that each processor has the same data
	MPI_Bcast(data,
			num_elements,
    		MPI_FLOAT,
    		0,
    		MPI_COMM_WORLD);
}

/*
* Divide the array into number of processor blocks
* Each block is of size (total number of elements/number of processors)
* Each processor is assigned to one block so that the computation can be parallelized across multiple nodes
*
* data: input array
* block_data: block array
* block_size: size of block_array
* rank: MPI rank
*/
void divide_data_block(float* data,
						float* block_data,
						const int block_size,
						const int rank)
{
	#pragma omp parallel for
	for (int i = 0; i < block_size; i++)
	{
		block_data[i] = data[rank*block_size+i];
	}
}

/*
* Initialize the GPU kernel for the scan function
* Copy data from the host to the device and then copy the output back to the host
* 
* data: input array
* scan_output: output scan array
* num_elements: size of input array
*/
void gpu_scan_kernel(float* data,
					float* scan_output, 
					int num_elements)
{
	int mem_size = sizeof(float) * num_elements;

	float *device_data, *device_scan_output;

	// allocate device memory input and output arrays
	checkCudaErrors(cudaMalloc((void**)&device_data, mem_size));
	checkCudaErrors(cudaMalloc((void**)&device_scan_output, mem_size));

	// copy host memory to device input array
	checkCudaErrors(cudaMemcpy(device_data, 
								data, 
								mem_size,
								cudaMemcpyHostToDevice));

	// execute the kernel
	int shared_mem_size = sizeof(float) * (num_elements+1);
	scan_gpu<<<1, num_elements, shared_mem_size>>>(device_scan_output, device_data);
	getLastCudaError("scan kernel execution failed");

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(scan_output, 
								device_scan_output, 
								mem_size,
								cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(device_data));
	checkCudaErrors(cudaFree(device_scan_output));
}

/*
* Add the scanned partial sum of the previous block onto the current block
* By adding the scanned partial sum, the current block will have the correct scan in the entire array
* 
* partial_sum: scanned partial sum of the block corresponding to the MPI rank
* data: input array
*/
__global__ void add_partial_sum(float partial_sum, 
								float* data)
{
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	data[tid] += partial_sum;
}

/*
* Initialize the GPU kernel for the partial sum function
* Copy data from the host to the device and then copy the output back to the host
* 
* partial_sum: scanned partial sum of the block corresponding to the MPI rank
* data: input array
* num_elements: size of input array
*/
void gpu_partial_sum_kernel(float partial_sum, 
							float* data, 
							int num_elements)
{
	int mem_size = sizeof(float) * num_elements;

	float *device_data;

	// allocate device memory input and output arrays
	checkCudaErrors(cudaMalloc((void**)&device_data, mem_size));

	// copy host memory to device input array
	checkCudaErrors(cudaMemcpy(device_data, 
								data, 
								mem_size,
								cudaMemcpyHostToDevice));

	// execute the kernel
	int shared_mem_size = sizeof(float) * (num_elements+1);
	add_partial_sum<<<1, num_elements, shared_mem_size>>>(partial_sum, device_data);
	getLastCudaError("scan kernel execution failed");

	// copy result from device to host
	checkCudaErrors(cudaMemcpy(data, 
								device_data, 
								mem_size,
								cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(device_data));
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[]) 
{
	// Time the run
	auto start = chrono::high_resolution_clock::now();

	MPI_Init(&argc, &argv);

	int rank, comm_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  	findCudaDevice(argc, (const char **)argv);

	// Parse command line arguments
	// random_data determines whether the generated array is made up of random or a constant number
	// check_output determines whether the scan output is printed and whether the GPU solution is checked against the CPU solution
	bool random_array = false;
	bool check_output = false;
	for (int i = 1; i < argc; i++) 
	{	
		string arg = argv[i];
		if (arg == "-random_data")
		{
			// Value is true or false
			istringstream(argv[i+1]) >> boolalpha >> random_array;
		}
		else if (arg == "-check_output")
		{
			// Value is true or false
			istringstream(argv[i+1]) >> boolalpha >> check_output;
		}
    }

	int num_elements = 512;
	float data[num_elements];
	create_data(data, 
				num_elements, 
				rank, 
				random_array);

	int block_size = num_elements/comm_size;
	float block_data[block_size]; 
	divide_data_block(data, 
					block_data, 
					block_size, 
					rank);
	assert(block_data[0] == data[rank*block_size]);

	float block_scan_output[block_size]; 
	gpu_scan_kernel(block_data, 
					block_scan_output,
					block_size);

	// Partial sum is the sum of a block, which is just the last element of the scan of the block
	// Gather the partial sum of each block onto the first processor
	float partial_sum[comm_size];
	MPI_Gather(&block_scan_output[block_size-1],
    			1,
    			MPI_FLOAT,
    			partial_sum,
    			1,
    			MPI_FLOAT,
    			0,
    			MPI_COMM_WORLD);

	// Compute a scan of the partial sum
	// Broadcast the partial sum array
	if (rank == 0)
	{
		gpu_scan_kernel(partial_sum, 
						partial_sum, 
						comm_size);
	}
	MPI_Bcast(partial_sum,
			comm_size,
			MPI_FLOAT,
			0,
			MPI_COMM_WORLD);

	printf("rank: %d, partial sum: %f\n", rank, partial_sum[rank]);

	gpu_partial_sum_kernel(partial_sum[rank-1], 
							block_scan_output, 
							block_size);
	
	// Gather the scan of each block to get the final scan output
	float scan_output[num_elements];
	MPI_Gather(&block_scan_output,
				block_size,
				MPI_FLOAT,
				scan_output,
				block_size,
				MPI_FLOAT,
				0,
				MPI_COMM_WORLD);

	if (rank == 0)
	{
		if (check_output)
		{	
			for (int i = 0; i < num_elements; i++)
			{
				printf("index: %d, scan: %f\n", i, scan_output[i]);
			}

			check_solution(scan_output,
							data, 
							num_elements);
		}
	}

	auto stop = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
	cout << "Time elapsed: " << duration.count() << endl;
	
	// CUDA exit -- needed to flush printf write buffer
	cudaDeviceReset();
	MPI_Finalize();
	return 0;
}
