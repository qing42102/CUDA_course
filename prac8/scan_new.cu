#include <string.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <chrono>
#include <tuple>
#include <vector>

#include <omp.h>
#include <mpi.h>
#include <helper_cuda.h>
#include <cublas_v2.h>
#include <cuda.h>

using namespace std;
#define MAX_THREADS 1024

/*
* CPU Scan
* 
* data: input array
* return the output scan array
*/
vector<float> scan_cpu(const vector<float> data) 
{
	vector<float> scan_output(data.size());

	scan_output[0] = data[0];
	for (int i = 1; i < data.size(); i++)
	{
		scan_output[i] = data[i] + scan_output[i-1];
	}

	return scan_output;
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
* Parse command line arguments
* 
* random_data determines whether the generated array is made up of random or a constant number
* check_output determines whether the scan output is printed and whether the GPU solution is checked against the CPU solution
* exponent determines the size of generated array, which is 2^exponent. 
*/
tuple<bool, bool, int> parse_arguments(int argc, char *argv[])
{
	// Default values
	bool random_array = false;
	bool check_output = false;
	int exponent = 3;

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
		else if (arg == "-num_elements")
		{
			exponent = atoi(argv[i+1]);
		}
    }

	tuple arguments = make_tuple(random_array, check_output, exponent);
	return arguments;
} 

/*
* Initialize the numbers for an array
* The initialization can be either random or a increasing number by 1
* 
* num_elements: size of generated array
* rank: MPI rank
* random: whether to use random or constant number for the generated array
* return generated array
*/
vector<float> create_data(const int num_elements, 
						const int rank, 
						const bool random)
{
	srand(0);

	vector<float> data(num_elements); 
	// Initialize only on the first processor
	if (rank == 0)
	{
		if (random)
		{
			#pragma omp parallel for
			for (int i = 0; i < num_elements; i++) 
			{
				data[i] = (rand() % 201) - 100;
			}
		}
		else
		{
			#pragma omp parallel for
			for (int i = 0; i < num_elements; i++) 
			{
				data[i] = i;
			}
		}
	}

	// Broadcast the data to all other processors to ensure that each processor has the same data
	MPI_Bcast(data.data(),
			num_elements,
    		MPI_FLOAT,
    		0,
    		MPI_COMM_WORLD);
	
	return data;
}

/*
* Divide the array into number of processor blocks
* Each block is of size (total number of elements/number of processors)
* Each processor is assigned to one block so that the computation can be parallelized across multiple nodes
*
* data: input array
* block_size: size of block_array
* rank: MPI rank
* return the block array
*/
vector<float> divide_data_block(vector<float> data,
								const int block_size,
								const int rank)
{
	vector<float> block_data(block_size);

	#pragma omp parallel for
	for (int i = 0; i < block_size; i++)
	{
		block_data[i] = data[rank*block_size+i];
	}

	return block_data;
}

/*
* Add a scalar elementwise to an array
* 
* data: input array. Note that the data should be on the GPU. 
* num_elements: size of the array
* constant: value to add to the array
*/
void scalar_vector_add(float* data,
						const int num_elements,
						const float constant)
{
    cublasHandle_t handle;
    cublasCreate(&handle);

	// Create vector of ones to get a*1 + y
	float *device_ones;
	checkCudaErrors(cudaMalloc((void **)&device_ones, sizeof(float) * num_elements));
	vector<float> ones(num_elements, 1.0);
	checkCudaErrors(cudaMemcpy(device_ones, 
								ones.data(), 
								sizeof(float) * num_elements,
								cudaMemcpyHostToDevice));

    cublasSaxpy_v2(handle, 
					num_elements, 
					&constant, 
					device_ones, 
					1, 
					data, 
					1);

	checkCudaErrors(cudaFree(device_ones));
	cublasDestroy(handle);
}

/*
* Initialize the GPU kernel for the scan function
* 
* data: input array
* scan_output: output scan array. Note that the returned output is on the device. 
*/
void gpu_scan_kernel(float* device_scan_output,
					vector<float> data)
{	
	int num_elements = data.size();
	int num_threads = min(MAX_THREADS, num_elements);
	int data_mem_size = sizeof(float) * num_threads;

	float *device_data;
	checkCudaErrors(cudaMalloc((void**)&device_data, data_mem_size));

	// Shared memory for storing scan results
	int shared_mem_size = sizeof(float) * (num_threads + 1);

	// If the number of elements is greater than the maximum threads for a GPU block, 
	// we need to divide the data into mutliple blocks
	// Each block's number of elements is the maximum threads
	if (num_elements > MAX_THREADS)
	{	
		// Initialize streams to parallelize the calculation of scan of each block
		int num_streams = num_elements / MAX_THREADS;
		cudaStream_t streams[num_streams];
		for (int i = 0; i < num_streams; i++)
		{
			cudaStreamCreate(&streams[i]);
		}

		vector<float> partial_sum(num_streams);
		for (int i = 0; i < num_streams; i++)
		{
			// Get the corresponding block of data for each stream
			vector<float> block_data(data.begin() + i*MAX_THREADS, 
									data.begin() + (i+1)*MAX_THREADS);

			checkCudaErrors(cudaMemcpy(device_data, 
										block_data.data(), 
										data_mem_size,
										cudaMemcpyHostToDevice));

			scan_gpu<<<1, num_threads, shared_mem_size, streams[i]>>>(device_scan_output + i*MAX_THREADS, device_data);

			// Partial sum is the sum of a block, which is just the last element of the scan of the block
			// Need to copy the element to host
			checkCudaErrors(cudaMemcpy(&partial_sum[i], 
										&device_scan_output[(i+1)*MAX_THREADS-1], 
										sizeof(float),
										cudaMemcpyDeviceToHost));
		}

		// Compute a scan of the partial sum
		vector<float> partial_sum_scan = scan_cpu(partial_sum);	
		
		// Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
		for (int i = 0; i < num_streams; i++)
		{
			scalar_vector_add(device_scan_output + i*MAX_THREADS, 
							MAX_THREADS, 
							partial_sum_scan[i-1]);

			cudaStreamDestroy(streams[i]);
		}
	}
	else
	{
		checkCudaErrors(cudaMemcpy(device_data, 
									data.data(), 
									data_mem_size,
									cudaMemcpyHostToDevice));

		scan_gpu<<<1, num_threads, shared_mem_size>>>(device_scan_output, device_data);
	}
	getLastCudaError("scan kernel execution failed");
	
	checkCudaErrors(cudaFree(device_data));
}

/*
* Calculate the scan of the partial sum of the scan of each block of data
* 
* block_scan_data: the scan of the block corresponding to the current rank. Note that this is on the GPU. 
* block_size: size of the block data
* comm_size: number of MPI processes 
* rank: MPI rank
*/
vector<float> partial_sum(float* block_scan_output, 
						int block_size, 
						int comm_size, 
						int rank)
{
	float last_element;
	checkCudaErrors(cudaMemcpy(&last_element, 
								&block_scan_output[block_size-1], 
								sizeof(float),
								cudaMemcpyDeviceToHost));

	// Partial sum is the sum of a block, which is just the last element of the scan of the block
	// Gather the partial sum of each block onto the first process
	vector<float> partial_sum(comm_size);
	MPI_Gather(&last_element,
    			1,
    			MPI_FLOAT,
    			partial_sum.data(),
    			1,
    			MPI_FLOAT,
    			0,
    			MPI_COMM_WORLD);

	// Compute a scan of the partial sum
	// Broadcast the partial sum scan array
	vector<float> partial_sum_scan(comm_size);
	if (rank == 0)
	{
		partial_sum_scan = scan_cpu(partial_sum);
	}
	MPI_Bcast(partial_sum_scan.data(),
			comm_size,
			MPI_FLOAT,
			0,
			MPI_COMM_WORLD);

	printf("rank: %d, partial sum: %f\n", rank, partial_sum_scan[rank]);

	return partial_sum_scan;
}

/*
* Check the GPU scan solution against the CPU scan solution
* Prints out the root mean squared difference
*
* scan_output: output scan array
* data: input array
*/
void check_solution(vector<float> scan_output, 
					vector<float> data)
{
	// compute reference solution by using CPU scan
	vector<float> reference = scan_cpu(data);

	float err = 0.0;
	for (int i = 0; i < scan_output.size(); i++) 
	{
		err += pow(scan_output[i] - reference[i], 2);
		printf("index: %d, GPU scan: %f, CPU scan: %f\n", i, scan_output[i], reference[i]);
	}
	printf("rms scan error = %f\n", sqrt(err/scan_output.size()));
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

	tuple<bool, bool, int> arguments = parse_arguments(argc, argv);
	bool random_array = get<0>(arguments);
	bool check_output = get<1>(arguments);
	int exponent = get<2>(arguments);

	int num_elements = pow(2, exponent);
	vector<float> data = create_data(num_elements, rank, random_array);

	int block_size = num_elements/comm_size;
	vector<float> block_data = divide_data_block(data, block_size, rank);
	assert(block_data[0] == data[rank*block_size]);

	float *block_scan_output; 
	checkCudaErrors(cudaMalloc((void**)&block_scan_output, sizeof(float) * block_size));
	gpu_scan_kernel(block_scan_output, block_data);

	vector<float> partial_sum_scan = partial_sum(block_scan_output, 
												block_size, 
												comm_size, 
												rank);
	
	// Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
	scalar_vector_add(block_scan_output, block_size, partial_sum_scan[rank-1]);
	
	float host_block_scan_output[block_size];
	checkCudaErrors(cudaMemcpy(host_block_scan_output, 
								block_scan_output, 
								sizeof(float) * block_size,
								cudaMemcpyDeviceToHost));

	// Gather the scan of each block to get the final scan output
	vector<float> scan_output(num_elements);
	MPI_Gather(&host_block_scan_output,
				block_size,
				MPI_FLOAT,
				scan_output.data(),
				block_size,
				MPI_FLOAT,
				0,
				MPI_COMM_WORLD);

	if (rank == 0)
	{
		if (check_output)
		{
			check_solution(scan_output, data);
		}

		printf("Last scan element: %f\n", scan_output[num_elements-1]);

		auto stop = chrono::high_resolution_clock::now();
		auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
		cout << "Time elapsed: " << duration.count() << endl;
	}
	
	checkCudaErrors(cudaFree(block_scan_output));

	// CUDA and MPI exit -- needed to flush printf write buffer
	cudaDeviceReset();
	MPI_Finalize();
	return 0;
}
