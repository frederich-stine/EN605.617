// Frederich Stine EN.605.617
// Module 5 Assignment

#include <stdio.h>
#include <stdint.h>

/******************* CUDA Kernel Prototypes ********************/
__global__
void gpu_all_arith_shared (int32_t* blockOne, int32_t* resultBlock);
__global__
void gpu_all_arith_shared_copy (int32_t* blockOne, int32_t* resultBlock);
__global__
void gpu_all_arith_const (int32_t* resultBlock);
__global__
void gpu_all_arith_only_const (int32_t* resultBlock);

/******************* Core Function Prototypes ********************/
void run_gpu_all_arith_shared (int op);
void run_gpu_all_arith_const (int op);

/******************* Global Variables ********************/

// Global variables used throughout file
uint32_t threadCount = 0;
uint32_t blockSize = 0;
uint32_t numBlocks = 0;
uint32_t arrSizeBytes = 0;

__constant__ int32_t const_arr[1024];
__constant__ int32_t value1 = 0x01234567;
__constant__ int32_t value2 = 0x89ABCDEF;
__constant__ int32_t value3 = 0x02468ACE;
__constant__ int32_t value4 = 0x13579BDF;

/******************* Funtion definitions ********************/
// Main function
// This function takes in command line arguments and invokes the
// correct core functions
int main(int argc, char** argv) {
	// Prints out a help menu if not enough params are passed
	if (argc != 4) {
		printf("Call ./assignment {numThreads} {blockSize} {operation}\n");
		printf("Operations: \n");
		printf("    0: Copy to shared Memory\n");
		printf("    1: Shared memory for local\n");
		printf("    2: Copy to constant memory\n");
		printf("    3: Constant memory only\n");
		exit(0);
	}

	// Load the parameters from the command line
	threadCount = atoi(argv[1]);
	blockSize = atoi(argv[2]);
	numBlocks = (threadCount+(blockSize-1))/blockSize;
	arrSizeBytes = threadCount*sizeof(int32_t);
	int operation = atoi(argv[3]);

	// Switch statement to call correct core function
	switch (operation) {
	case 0:
	case 1:
		run_gpu_all_arith_shared(operation);
		break;
	case 2:
	case 3:
		run_gpu_all_arith_const(operation-2);
		break;
	default:
		printf("Incorrect operation specified: %d", operation);
		exit(0);
	}
}

void run_gpu_all_arith_shared (int op) {
	int32_t *one, *result;
	int32_t *d_one, *d_result;

	// Allocated page locked  memory using standard c malloc function
	cudaMallocHost((void**)&one, 1024*4);
	cudaMallocHost((void**)&result, 1024*4);

	// Initialize memory - general initialization
	for(int i=0; i<1024; i++) {
		one[i] = i;
	}
	
	// Allocate memroy on the GPU for computation
	cudaMalloc((void**)&d_one, 1024*4);
	cudaMalloc((void**)&d_result, 1024*4);

	// Copy memory from host to GPU - pinned memory
	cudaMemcpy(d_one, one, 1024*4, cudaMemcpyHostToDevice);

	// Run kernel
	switch (op) {
		case 0:
			gpu_all_arith_shared<<<numBlocks, blockSize, blockSize*4>>>(d_one, d_result);
			break;
		case 1:
			gpu_all_arith_shared_copy<<<numBlocks, blockSize, blockSize*8>>>(d_one, d_result);
			break;
	}

	// Copy memory back from GPU to host
	cudaMemcpy(result, d_result, 1024*4, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_result);

	// Free pinned memory on host
	cudaFreeHost(one);
	cudaFreeHost(result);
}

void run_gpu_all_arith_const (int op) {
	int32_t *result, *one;
	int32_t *d_result;

	// Allocated page locked  memory using standard c malloc function
	one = (int32_t*)malloc(1024*4);
	cudaMallocHost((void**)&result, 1024*4);

	// Initialize memory - general initialization
	for(int i=0; i<1024; i++) {
		result[i] = 0;
		one[i] = i;
	}
	
	// Allocate memory on the GPU for computation
	cudaMalloc((void**)&d_result, 1024*4);

	// Copy memory from host to GPU - pinned memory
	cudaMemcpy(d_result, result, 1024*4, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(const_arr, one, 1024*4);

	// Run kernel
	switch (op) {
		case 0:
			gpu_all_arith_const<<<4, 256>>>(d_result);
			break;
		case 1:
			gpu_all_arith_only_const<<<4, 256>>>(d_result);
			break;
	}

	// Copy memory back from GPU to host
	cudaMemcpy(result, d_result, 1024*4, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(d_result);

	// Free pinned memory on host
	free(one);
	cudaFreeHost(result);
}

__global__
void gpu_all_arith_shared (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	extern __shared__ int32_t s[];

	s[threadIdx.x] = 0;
	s[threadIdx.x] += blockOne[thread_idx];
	s[threadIdx.x] *= blockOne[thread_idx];
	s[threadIdx.x] -= blockOne[thread_idx];
	s[threadIdx.x] %= blockOne[thread_idx];
	
	resultBlock[thread_idx] = s[threadIdx.x];
}

__global__
void gpu_all_arith_shared_copy (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	extern __shared__ int32_t s[];

	// Copy to shared memory
	s[threadIdx.x] = blockOne[thread_idx];

	// Execute
	s[threadIdx.x+blockDim.x] = 0;
	s[threadIdx.x+blockDim.x] += blockOne[threadIdx.x];
	s[threadIdx.x+blockDim.x] *= blockOne[threadIdx.x];
	s[threadIdx.x+blockDim.x] -= blockOne[threadIdx.x];
	s[threadIdx.x+blockDim.x] %= blockOne[threadIdx.x];
	
	// Copy back to global
	resultBlock[thread_idx] = s[threadIdx.x+blockDim.x];
}

__global__
void gpu_all_arith_const (int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	resultBlock[thread_idx] += const_arr[thread_idx];
	resultBlock[thread_idx] *= const_arr[thread_idx];
	resultBlock[thread_idx] /= const_arr[thread_idx];
	resultBlock[thread_idx] -= const_arr[thread_idx];
}

__global__
void gpu_all_arith_only_const (int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	resultBlock[thread_idx] += value1;
	resultBlock[thread_idx] *= value2;
	resultBlock[thread_idx] /= value3;
	resultBlock[thread_idx] -= value4;
}
