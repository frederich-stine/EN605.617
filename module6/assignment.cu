// Frederich Stine EN.605.617
// Module 6 Assignment

#include <stdio.h>
#include <stdint.h>

/******************* CUDA Kernel Prototypes ********************/
__global__
void gpu_register_copy_arith (int32_t* blockOne, int32_t* resultBlock);
__global__
void gpu_register_arith (int32_t* blockOne, int32_t* resultBlock);
__global__
void gpu_global_arith (int32_t* blockOne, int32_t* resultBlock);

/******************* Core Function Prototypes ********************/
void run_all_arith (int op);

/******************* Global Variables ********************/
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
		printf("Call ./assignment {threadCount} {blockSize} {operation}\n");
		printf("Operations: \n");
		printf("    0: Copy to register\n");
		printf("    1: Register local variables\n");
		printf("    2: Global memory only\n");
		exit(0);
	}

	// Load the parameters from the command line
	threadCount = atoi(argv[1]);
	blockSize = atoi(argv[2]);
	numBlocks = (threadCount+(blockSize-1))/blockSize;
	arrSizeBytes = threadCount*sizeof(int32_t);
	int operation = atoi(argv[3]);

	// Run the kernel
	run_all_arith(operation);
}

void run_all_arith (int op) {
	int32_t *one, *result;
	int32_t *d_one, *d_result;

	// Allocated page locked  memory using standard c malloc function
	cudaMallocHost((void**)&one, arrSizeBytes);
	cudaMallocHost((void**)&result, arrSizeBytes);

	for(int i=0; i<1024; i++) {
		one[i] = i;
	}

	// Allocate memroy on the GPU for computation
	cudaMalloc((void**)&d_one, arrSizeBytes);
	cudaMalloc((void**)&d_result, arrSizeBytes);

	// Copy memory from host to GPU - pinned memory
	cudaMemcpy(d_one, one, arrSizeBytes, cudaMemcpyHostToDevice);

	// Run kernel
	switch (op) {
		case 0:
			gpu_register_copy_arith<<<numBlocks, blockSize>>>(d_one, d_result);
			break;
		case 1:
			gpu_register_arith<<<numBlocks, blockSize>>>(d_one, d_result);
			break;
		case 2:
			gpu_global_arith<<<numBlocks, blockSize>>>(d_one, d_result);
			break;
	}

	// Copy memory back from GPU to host
	cudaMemcpy(result, d_result, arrSizeBytes, cudaMemcpyDeviceToHost);
	
	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_result);

	// Free pinned memory on host
	cudaFreeHost(one);
	cudaFreeHost(result);
}

__global__
void gpu_register_copy_arith (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int32_t localRegOne, localRegResult;
	localRegOne = blockOne[thread_idx];
	localRegResult = 0;

	for (int i=0; i<10000; i++) {
		localRegResult += localRegOne;
		localRegResult *= localRegOne;
		localRegResult -= localRegOne;
		localRegResult -= localRegOne;
	}

	// Copy back to global mem
	resultBlock[thread_idx] = localRegResult;
}


__global__
void gpu_register_arith (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	int32_t localRegResult;
	localRegResult = 0;

	for (int i=0; i<10000; i++) {
		localRegResult += blockOne[thread_idx];
		localRegResult *= blockOne[thread_idx];
		localRegResult -= blockOne[thread_idx];
		localRegResult /= blockOne[thread_idx];
	}

	// Copy back to global mem
	resultBlock[thread_idx] = localRegResult;
}


__global__
void gpu_global_arith (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	resultBlock[thread_idx] = 0;

	for (int i=0; i<10000; i++) {
		resultBlock[thread_idx] += blockOne[thread_idx];
		resultBlock[thread_idx] *= blockOne[thread_idx];
		resultBlock[thread_idx] -= blockOne[thread_idx];
		resultBlock[thread_idx] /= blockOne[thread_idx];
	}
}
