#include <string.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <chrono>
#include <tuple>
#include <vector>
#include <random>
#include <stdint.h>

#include <omp.h>
#include <mpi.h>
#include <helper_cuda.h>
#include <cuda.h>

using namespace std;
#define MAX_THREADS 1024
#define MAX_SCAN_BLOCK 2048

/*
* CPU Scan
* 
* data: input array
*
* return the output scan array
*/
vector<float> scan_cpu(const vector<float> &data) 
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
* GPU Scan with Blelloch algorithm
* 
* scan_output: output scan array
* data: input array
* num_elements: size of array
*/
__global__ void scan_gpu(float *g_odata, 
                        float *g_idata, 
                        const unsigned int num_elements)
{
    extern __shared__ float temp[];  // allocated on invocation

    int thid = threadIdx.x;
    int offset = 1;

    temp[2*thid] = g_idata[2*thid]; // load input into shared memory
    temp[2*thid+1] = g_idata[2*thid+1];

    for (int d = num_elements>>1; d > 0; d >>= 1) // build sum in place up the tree
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

    if (thid == 0) { temp[num_elements - 1] = 0; } // clear the last element
    
    for (int d = 1; d < num_elements; d *= 2) // traverse down tree & build scan
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
    unsigned int exponent = 3;

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
            assert(exponent > 0);
        }
    }

    tuple<bool, bool, int> arguments = make_tuple(random_array, check_output, exponent);
    return arguments;
} 

/*
* Initialize the numbers for an array
* The initialization can be either random or a increasing number by 1
* Divide the array into number of processor blocks
* Each processor is assigned to one block so that the computation can be parallelized across multiple nodes
* 
* num_elements: size of generated array
* random: whether to use random or constant number for the generated array
* block_size: size of each block
* rank: MPI rank
* return_whole_data: choose whether to return the entire data rather than the block data
* 
* return generated array
*/
vector<float> create_data(const uint64_t num_elements, 
                        const unsigned int block_size, 
                        const int rank,
                        const bool return_whole_data, 
                        const bool random)
{
    vector<float> data; 
    // Intialize only on the first rank to save memory
    if (rank == 0)
    {
        data.resize(num_elements);

        if (random)
        {
            #pragma omp parallel
            {
                // Thread-safe method to generate the same numbers across threads
                uniform_int_distribution<int> distribution(-20, 20);
                mt19937 engine(0);

                #pragma omp for
                for (int i = 0; i < num_elements; i++) 
                {
                    data[i] = float(distribution(engine));
                }
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

    if (return_whole_data)
    {
        return data;
    }
    else
    {
        // Each block is of size (total number of elements/number of processors)
        vector<float> block_data(block_size);
        MPI_Scatter(data.data(),
                    block_size,
                    MPI_FLOAT,
                    block_data.data(),
                    block_size,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD);
                    
        return block_data;
    }
}

/*
* Add a scalar elementwise to an array
* Faster than cublasSaxpy
* 
* vec: input array.  
* num_elements: size of the array
* constant: value to add to the array
*/
__global__ void scalar_vector_add(float* vec,
                                const unsigned int num_elements,
                                const float constant)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    // For adding to the specified part of the array
    if (tid < num_elements)
    {
        vec[tid] += constant;
    }
}

/*
* Add two vectors
* Faster than cublasSaxpy
* 
* vec1: input array. 
* vec2: input array
* num_elements: size of the array
*/
__global__ void vector_addition(float* vec1,
                                float* vec2, 
                                const unsigned int num_elements)
{
    int tid = threadIdx.x + blockDim.x*blockIdx.x;
    // For adding to the specified part of the array
    if (tid < num_elements)
    {
        vec2[tid] += vec1[tid];
    }
}

/*
* Initialize the GPU kernel for the scan function
* 
* data: input array
* scan_output: output scan array. Note that the returned output is on the device. 
*/
void gpu_scan_kernel(float* device_scan_output,
                    const vector<float> &data)
{
    unsigned int num_elements = data.size();

    unsigned int data_mem_size = sizeof(float) * num_elements;
    float *device_data;
    checkCudaErrors(cudaMalloc((void**)&device_data, data_mem_size));

    checkCudaErrors(cudaMemcpy(device_data, 
                                data.data(), 
                                data_mem_size,
                                cudaMemcpyHostToDevice));

    // Shared memory for storing scan results
    unsigned int shared_mem_size = sizeof(float) * (num_elements + 1);

    scan_gpu<<<1, num_elements/2, shared_mem_size>>>(device_scan_output, 
                                                    device_data, 
                                                    num_elements);

    // Turn exclusive scan into inclusive scan
    vector_addition<<<1, num_elements>>>(device_data, 
                                        device_scan_output, 
                                        num_elements);

    checkCudaErrors(cudaFree(device_data));
    getLastCudaError("scan kernel execution failed");
}

/*
* GPU kernel for if the number of elements is greater than the maximum threads for a GPU block
* Divide the data into mutliple blocks
* Each block's size is the maximum threads
* 
* data: input array
* scan_output: output scan array. Note that the returned output is on the device. 
*/
void gpu_block_scan_kernel(float* device_scan_output,
                            const vector<float> &data)
{
    unsigned int num_elements = data.size();

    unsigned int data_mem_size = sizeof(float) * MAX_SCAN_BLOCK;

    // Shared memory for storing scan results
    unsigned int shared_mem_size = sizeof(float) * (MAX_SCAN_BLOCK + 1);

    // Initialize streams to parallelize the scan of each block
    int num_streams = num_elements / MAX_SCAN_BLOCK;
    cudaStream_t streams[num_streams];

    vector<float> partial_sum(num_streams);

    // Create a 2D array for storing the scan results for each stream or block
    float *block_scan_output[num_streams];

    for (int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);

        checkCudaErrors(cudaMalloc(&block_scan_output[i], data_mem_size));

        float *device_data;
        checkCudaErrors(cudaMalloc((void**)&device_data, data_mem_size));

        // Get the corresponding block of data for each stream
        vector<float> block_data(data.begin() + i*MAX_SCAN_BLOCK, 
                                data.begin() + (i+1)*MAX_SCAN_BLOCK);

        checkCudaErrors(cudaMemcpyAsync(device_data, 
                                        block_data.data(), 
                                        data_mem_size,
                                        cudaMemcpyHostToDevice,
                                        streams[i]));

        scan_gpu<<<1, MAX_THREADS, shared_mem_size, streams[i]>>>(block_scan_output[i], 
                                                                device_data, 
                                                                MAX_SCAN_BLOCK);
        
        // Turn exclusive scan into inclusive scan
        vector_addition<<<2, MAX_THREADS, 0, streams[i]>>>(device_data, 
                                                            block_scan_output[i], 
                                                            MAX_SCAN_BLOCK);

        // Partial sum is the sum of a block, which is just the last element of the scan of the block
        // Need to copy the element to host
        checkCudaErrors(cudaMemcpyAsync(&partial_sum[i], 
                                        &block_scan_output[i][MAX_SCAN_BLOCK-1], 
                                        sizeof(float),
                                        cudaMemcpyDeviceToHost, 
                                        streams[i]));

        checkCudaErrors(cudaFree(device_data));
    }

    getLastCudaError("scan kernel execution failed");
    
    // Compute a scan of the partial sum
    vector<float> partial_sum_scan = scan_cpu(partial_sum);
    
    // Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
    for (int i = 1; i < num_streams; i++)
    {
        scalar_vector_add<<<2, MAX_THREADS, 0, streams[i]>>>(block_scan_output[i], 
                                                            MAX_SCAN_BLOCK, 
                                                            partial_sum_scan[i-1]);
    }
    
    for (int i = 0; i < num_streams; i++)
    {
        // Copy/flatten the results into the output array
        checkCudaErrors(cudaMemcpy(device_scan_output + i*MAX_SCAN_BLOCK, 
                                    block_scan_output[i], 
                                    data_mem_size,
                                    cudaMemcpyDeviceToDevice));

        cudaStreamDestroy(streams[i]);
    }
    
}

/*
* Calculate the scan of the partial sum of the scan of each block of data
* The partial sum is the last element of the scan on each MPI process
* 
* block_scan_data: the scan of the block corresponding to the current rank. Note that this is on the GPU. 
* block_size: size of the block data
* comm_size: number of MPI processes 
* rank: MPI rank

* return the scan of the partial sum
*/
vector<float> partial_sum(float* block_scan_output, 
                        const unsigned int block_size, 
                        const int comm_size, 
                        const int rank)
{
    float last_element;
    checkCudaErrors(cudaMemcpy(&last_element, 
                                &block_scan_output[block_size-1], 
                                sizeof(float),
                                cudaMemcpyDeviceToHost));

    // Partial sum is the sum of a block, which is just the last element of the scan of the block
    // Gather the partial sum of each block onto the first process
    vector<float> partial_sum(comm_size);
    MPI_Allgather(&last_element,
                1,
                MPI_FLOAT,
                partial_sum.data(),
                1,
                MPI_FLOAT, 
                MPI_COMM_WORLD);

    // Compute a scan of the partial sum
    // Broadcast the partial sum scan array
    vector<float> partial_sum_scan(comm_size);
    partial_sum_scan = scan_cpu(partial_sum);

    return partial_sum_scan;
}

/*
* Check the GPU scan solution against the CPU scan solution
* Prints out the root mean squared difference
*
* scan_output: output scan array
* data: input array
*/
void check_solution(vector<float> &scan_output, 
                    vector<float> &data)
{
    // compute reference solution by using CPU scan
    vector<float> reference = scan_cpu(data);

    float err = 0.0;
    for (int i = 0; i < scan_output.size(); i++) 
    {
        err += pow(scan_output[i] - reference[i], 2);
        // printf("index: %d, GPU scan: %f, CPU scan: %f\n", i, scan_output[i], reference[i]);
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

    int num_GPU;
    cudaGetDeviceCount(&num_GPU);
    cudaSetDevice(rank / (float(comm_size) / float(num_GPU)));

    tuple<bool, bool, int> arguments = parse_arguments(argc, argv);
    bool random_array = get<0>(arguments);
    bool check_output = get<1>(arguments);
    unsigned int exponent = get<2>(arguments);

    uint64_t num_elements = pow(2, exponent);
    unsigned int block_size = num_elements/comm_size;
    assert(block_size > 1);
    vector<float> block_data = create_data(num_elements, 
                                            block_size, 
                                            rank,
                                            false, 
                                            random_array);

    if (rank == 0)
    {
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Finished creating data " << "Time elapsed: " << duration.count() << endl;
    }

    float *block_scan_output; 
    checkCudaErrors(cudaMalloc((void**)&block_scan_output, sizeof(float) * block_size));
    if (block_size > MAX_SCAN_BLOCK)
    {
        gpu_block_scan_kernel(block_scan_output, block_data);
    }
    else
    {
        gpu_scan_kernel(block_scan_output, block_data);
    }

    if (rank == 0)
    {
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Finished GPU scan " << "Time elapsed: " << duration.count() << endl;
    }

    vector<float> partial_sum_scan = partial_sum(block_scan_output, 
                                                block_size, 
                                                comm_size, 
                                                rank);

    if (check_output)
    {
        printf("rank: %d, partial sum: %f\n", rank, partial_sum_scan[rank]);
    }
    
    // Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
    int num_gpu_block = max(1, block_size / MAX_THREADS);
    scalar_vector_add<<<num_gpu_block, MAX_THREADS>>>(block_scan_output, 
                                                    block_size, 
                                                    partial_sum_scan[rank-1]);

    vector<float> host_block_scan_output(block_size);
    checkCudaErrors(cudaMemcpy(host_block_scan_output.data(), 
                                block_scan_output, 
                                sizeof(float) * block_size,
                                cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(block_scan_output));

    // Gather the scan of each block to get the final scan output
    // Initialize the output vector only rank 0 to save memory
    vector<float> scan_output;
    if (rank == 0)
    {
        scan_output.resize(num_elements);
    }
    MPI_Gather(host_block_scan_output.data(),
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
            // Get the whole data for checking with CPU scan
            vector<float> data = create_data(num_elements, 
                                            block_size, 
                                            rank,
                                            true, 
                                            random_array);

            check_solution(scan_output, data);
        }

        printf("Last scan element: %f\n", scan_output[num_elements-1]);

        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
        cout << "Time elapsed: " << duration.count() << endl;
    }

    // CUDA and MPI exit -- needed to flush printf write buffer
    cudaDeviceReset();
    MPI_Finalize();
    return 0;
}
