// Frederich Stine EN.605.617
// Module 4 Assignment

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/******************* CUDA Function Prototypes ********************/
// This CUDA kernel applies all four arithmetic functions from the
// previous assignment
__global__
void gpu_all_arith (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock);

/******************* Core Function Prototypes ********************/
// Function prototype for core function to run with paged mem
void run_paged_mem(void);
// Function prototype for core function to run with pinned mem
void run_pinned_mem(void);

// Global variables used throughout file
uint32_t threadCount = 0;
uint32_t blockSize = 0;
uint32_t numBlocks = 0;
uint32_t arrSizeBytes = 0;

/******************* Funtion definitions ********************/
// Main function
// This function takes in command line arguments and invokes the
// correct core functions
int main(int argc, char** argv) {
	// Prints out a help menu if not enough params are passed
	if (argc != 4) {
		printf("Call ./assignment {numThreads} {blockSize} {operation}\n");
		printf("Operations: \n");
		printf("    0: Paged Memory\n");
		printf("    1: Pinned Memory\n");
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
		run_paged_mem();
		break;
	case 1:
		run_pinned_mem();
		break;
	default:
		printf("Incorrect operation specified: %d", operation);
		exit(0);
	}
}

// This function runs the all_arith kernel with the thread count and
// block size specified using paged memory on the host
void run_paged_mem (void) {
	int32_t *one, *two, *result;
	int32_t *d_one, *d_two, *d_result;

	// Allocated paged memory using standard c malloc function
	one = (int32_t*)malloc(arrSizeBytes);
	two = (int32_t*)malloc(arrSizeBytes);
	result = (int32_t*)malloc(arrSizeBytes);

	// Initialize memory - general initialization
	for(int i=0; i<threadCount; i++) {
		one[i] = i;
		two[i] = threadCount-i;
	}

	// Allocate memory on the GPU for computation
	cudaMalloc((void**)&d_one, arrSizeBytes);
	cudaMalloc((void**)&d_two, arrSizeBytes);
	cudaMalloc((void**)&d_result, arrSizeBytes);

	// Copy memory from host to GPU - paged memory
	cudaMemcpy(d_one, one, arrSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_two, two, arrSizeBytes, cudaMemcpyHostToDevice);

	// Run kernel
	gpu_all_arith<<<numBlocks, blockSize>>>(d_one, d_two, d_result);

	// Copy memory back from GPU to host
	cudaMemcpy(result, d_result, arrSizeBytes, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_two);
	cudaFree(d_result);

	// Free memory on host
	free(one);
	free(two);
	free(result);
}

// This function runs the all_arith kernel with the thread count and
// bock size specificed using page locked memory on the host
void run_pinned_mem (void) {
	int32_t *one, *two, *result;
	int32_t *d_one, *d_two, *d_result;

	// Allocated page locked  memory using standard c malloc function
	cudaMallocHost((void**)&one, arrSizeBytes);
	cudaMallocHost((void**)&two, arrSizeBytes);
	cudaMallocHost((void**)&result, arrSizeBytes);

	// Initialize memory - general initialization
	for(int i=0; i<threadCount; i++) {
		one[i] = i;
		two[i] = threadCount-i;
	}
	
	// Allocate memroy on the GPU for computation
	cudaMalloc((void**)&d_one, arrSizeBytes);
	cudaMalloc((void**)&d_two, arrSizeBytes);
	cudaMalloc((void**)&d_result, arrSizeBytes);

	// Copy memory from host to GPU - pinned memory
	cudaMemcpy(d_one, one, arrSizeBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_two, two, arrSizeBytes, cudaMemcpyHostToDevice);

	// Run kernel
	gpu_all_arith<<<numBlocks, blockSize>>>(d_one, d_two, d_result);

	// Copy memory back from GPU to host
	cudaMemcpy(result, d_result, arrSizeBytes, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_two);
	cudaFree(d_result);

	// Free pinned memory on host
	cudaFreeHost(one);
	cudaFreeHost(two);
	cudaFreeHost(result);
}


// GPU all arith kernel
// Adds, mults, subs, and mods the values at the first two threads
// into the result thread
__global__
void gpu_all_arith (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = blockOne[thread_idx] + blockTwo[thread_idx];
	resultBlock[thread_idx] = blockOne[thread_idx] * blockTwo[thread_idx];
	resultBlock[thread_idx] = blockOne[thread_idx] - blockTwo[thread_idx];
	resultBlock[thread_idx] = blockOne[thread_idx] % blockTwo[thread_idx];
}
