#include <string.h>
#include <math.h>
#include <iostream>
#include <assert.h>
#include <sstream>
#include <chrono>
#include <tuple>
#include <vector>
#include <random>

#include <omp.h>
#include <mpi.h>
#include <helper_cuda.h>
#include <cuda.h>

using namespace std;
#define MAX_THREADS 2048

/*
* CPU Scan
* 
* data: input array
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
                        const int num_elements)
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
* 
* num_elements: size of generated array
* random: whether to use random or constant number for the generated array
* return generated array
*/
vector<float> create_data(const int num_elements, 
                        const bool random)
{
    vector<float> data(num_elements); 
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
    
    return data;
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
                                const int num_elements,
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
                                const int num_elements)
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
                    vector<float> &data)
{			
    auto start = chrono::high_resolution_clock::now();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned int num_elements = data.size();
    int num_threads = min(MAX_THREADS, num_elements);
    unsigned int data_mem_size = sizeof(float) * num_threads;

    // Shared memory for storing scan results
    unsigned int shared_mem_size = sizeof(float) * (num_threads + 1);

    // If the number of elements is greater than the maximum threads for a GPU block, 
    // we need to divide the data into mutliple blocks
    // Each block's number of elements is the maximum threads
    if (num_elements > MAX_THREADS)
    {	
        // Initialize streams to parallelize the calculation of scan of each block
        int num_streams = num_elements / MAX_THREADS;
        cudaStream_t streams[num_streams];

        vector<float> partial_sum(num_streams);

        for (int i = 0; i < num_streams; i++)
        {
            cudaStreamCreate(&streams[i]);

            float *device_data;
            checkCudaErrors(cudaMalloc((void**)&device_data, data_mem_size));

            // Get the corresponding block of data for each stream
            vector<float> block_data(data.begin() + i*MAX_THREADS, 
                                    data.begin() + (i+1)*MAX_THREADS);

            checkCudaErrors(cudaMemcpyAsync(device_data, 
                                            block_data.data(), 
                                            data_mem_size,
                                            cudaMemcpyHostToDevice,
                                            streams[i]));

            scan_gpu<<<1, MAX_THREADS/2, shared_mem_size, streams[i]>>>(device_scan_output + i*MAX_THREADS, 
                                                                        device_data, 
                                                                        MAX_THREADS);

            cudaStreamSynchronize(streams[i]);
            
            // Turn exclusive scan into inclusive scan
            vector_addition<<<2, MAX_THREADS/2, 0, streams[i]>>>(device_data, 
                                                                device_scan_output + i*MAX_THREADS, 
                                                                MAX_THREADS);

            cudaStreamSynchronize(streams[i]);

            // Partial sum is the sum of a block, which is just the last element of the scan of the block
            // Need to copy the element to host
            checkCudaErrors(cudaMemcpy(&partial_sum[i], 
                                        &device_scan_output[(i+1)*MAX_THREADS-1], 
                                        sizeof(float),
                                        cudaMemcpyDeviceToHost));

            checkCudaErrors(cudaFree(device_data));
        }

        if (rank == 0)
        {
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
            cout << "Finished GPU scan " << "Time elapsed: " << duration.count() << endl;
            start = chrono::high_resolution_clock::now();
        }
        
        // Compute a scan of the partial sum
        vector<float> partial_sum_scan = scan_cpu(partial_sum);
        
        // Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
        for (int i = 1; i < num_streams; i++)
        {	
            scalar_vector_add<<<2, MAX_THREADS/2>>>(device_scan_output + i*MAX_THREADS, 
                                                    MAX_THREADS, 
                                                    partial_sum_scan[i-1]);
        }

        if (rank == 0)
        {
            auto stop = chrono::high_resolution_clock::now();
            auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
            cout << "Finished adding partial sum to scan " << "Time elapsed: " << duration.count() << endl;
        }
    }
    else
    {
        float *device_data;
        checkCudaErrors(cudaMalloc((void**)&device_data, data_mem_size));

        checkCudaErrors(cudaMemcpy(device_data, 
                                    data.data(), 
                                    data_mem_size,
                                    cudaMemcpyHostToDevice));

        scan_gpu<<<1, num_threads/2, shared_mem_size>>>(device_scan_output, 
                                                        device_data, 
                                                        num_elements);

        // Turn exclusive scan into inclusive scan
        vector_addition<<<1, num_elements>>>(device_data, 
                                            device_scan_output, 
                                            num_elements);

        checkCudaErrors(cudaFree(device_data));
    }
    getLastCudaError("scan kernel execution failed");
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

    unsigned int num_elements = pow(2, exponent);
    vector<float> data = create_data(num_elements, random_array);

    // Divide the array into number of processor blocks
    // Each block is of size (total number of elements/number of processors)
    // Each processor is assigned to one block so that the computation can be parallelized across multiple nodes
    unsigned int block_size = num_elements/comm_size;
    vector<float> block_data(data.begin() + rank*block_size, 
                            data.begin() + (rank+1)*block_size);

    float *block_scan_output; 
    checkCudaErrors(cudaMalloc((void**)&block_scan_output, sizeof(float) * block_size));
    gpu_scan_kernel(block_scan_output, block_data);

    vector<float> partial_sum_scan = partial_sum(block_scan_output, 
                                                block_size, 
                                                comm_size, 
                                                rank);
    
    // Add the scan of the partial sum to the corresponding scan of each block to get the correct overall scan
    scalar_vector_add<<<max(1, block_size / 1024), 1024>>>(block_scan_output, 
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
