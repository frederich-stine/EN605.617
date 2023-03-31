// Frederich Stine EN.605.617
// Module 3 Assignment
// This program incorporates all requirements of the module 3 assignment
// This includes Cuda add, mult, sub, and mod as well as conditional
// branch testing.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

/******************* CUDA Function Prototypes ********************/
// Function prototype for add kernel
__global__
void gpu_add (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock);
// Function prototype for sub kernel
__global__
void gpu_sub (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock);
// Function prototype for mult kernel
__global__
void gpu_mult (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock);
// Function prototype for mod kernel
__global__
void gpu_mod (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock);
// Function prototype for branch test kernel
__global__
void gpu_branch (int32_t* blockOne, int32_t* resultBlock);
// Function prototype for non-branch test kernel
__global__
void gpu_no_branch (int32_t* blockOne, int32_t* resultBlock);

/******************* Core Function Prototypes ********************/
// This function invokes the four arithmetic kernels based on the 
// op parameter.
void run_arith(int op);
// This function runs the branch_comparison test code
void run_branch_compare(void);

/******************* Helper Function Prototypes ********************/
// This function does conditional branch testing on the CPU
void cpu_branch (int32_t* arrayOne, int32_t* resultArray);
// This function prints out the blocks in an easily readable format
void print_blocks (int32_t* resultArr);

// Global variables used throughout file
uint32_t threadCount = 0;
uint32_t blockSize = 0;
uint32_t numBlocks = 0;
uint32_t gpuArrSize = 0;

/******************* Funtion definitions ********************/
// Main function
// This function takes in command line arguments and invokes the
// correct core functions
int main(int argc, char** argv) {
	// Prints out a help menu if not enough params are passed
	if (argc != 4) {
		printf("Call ./assignment {numThreads} {blockSize} {operation}\n");
		printf("Operations: \n");
		printf("    0: Add\n");
		printf("    1: Sub\n");
		printf("    2: Mult\n");
		printf("    3: Mod\n");
		printf("    4: Branch Compare\n");
		exit(0);
	}

	// Load the parameters from the command line
	threadCount = atoi(argv[1]);
	blockSize = atoi(argv[2]);
	numBlocks = threadCount/blockSize;
	gpuArrSize = threadCount*sizeof(int32_t);

	int operation = atoi(argv[3]);

	// Switch statement to call correct core function
	switch (operation) {
	case 0:
	case 1:
	case 2:
	case 3:
		run_arith(operation);
		break;
	case 4:
		run_branch_compare();
		break;
	default:
		printf("Incorrect operation specified: %d", operation);
		exit(0);
	}
}

// Fun arith function
// This function sets up arrays of data as specified in the handout
// This funtion then invokes the correct arithmetic kernel after
// allocation memory on the GPU and finally copies the result back and
// prints it out to the user.
void run_arith (int op) {
	// Initialize arrays for data
	int32_t arrayOne[threadCount];
	int32_t arrayTwo[threadCount];
	int32_t arrayResult[threadCount];

	srand(time(NULL));
	// Init arrays
	for (int32_t i=0; i<threadCount; i++) {
		arrayOne[i] = i;
		arrayTwo[i] = rand() % 4;
	}

	// Allocate data on the GPU
	int32_t* gpuArrayOne;
	int32_t* gpuArrayTwo;
	int32_t* gpuResult;
	cudaMalloc((void**)&gpuArrayOne, gpuArrSize);
	cudaMalloc((void**)&gpuArrayTwo, gpuArrSize);
	cudaMalloc((void**)&gpuResult, gpuArrSize);
	
	// Copy data to the GPU
	cudaMemcpy(gpuArrayOne, arrayOne, gpuArrSize, cudaMemcpyHostToDevice); 
	cudaMemcpy(gpuArrayTwo, arrayTwo,  gpuArrSize, cudaMemcpyHostToDevice); 
	
	// Run correct arithmetic kernel
	switch (op) {
	case 0:
		gpu_add<<<numBlocks, blockSize>>>(gpuArrayOne, gpuArrayTwo, gpuResult);
		break;
	case 1:
		gpu_sub<<<numBlocks, blockSize>>>(gpuArrayOne, gpuArrayTwo, gpuResult);
		break;
	case 2:
		gpu_mult<<<numBlocks, blockSize>>>(gpuArrayOne, gpuArrayTwo, gpuResult);
		break;
	case 3:
		gpu_mod<<<numBlocks, blockSize>>>(gpuArrayOne, gpuArrayTwo, gpuResult);
		break;
	}
	
	// Copy the result back from the GPU
	cudaMemcpy(arrayResult, gpuResult, gpuArrSize, cudaMemcpyDeviceToHost); 
	
	// Print the result to the user
	printf("Result: \n");
	print_blocks(arrayResult);
}

// Run branch compare function
// This function runs a comparison between conditional branching code
// on the GPU vs the CPU vs a non-branching test case.
// This is compared through the time needed to execute.
void run_branch_compare (void) {
	// Initialize arrays
	int32_t arrayOne[threadCount];
	int32_t arrayResult[threadCount];

	for (int32_t i=0; i<threadCount; i++) {
		arrayOne[i] = i;
	}

	// Allocate arrays on GPU
	int32_t* gpuArrayOne;
	int32_t* gpuResult;
	cudaMalloc((void**)&gpuArrayOne, gpuArrSize);
	cudaMalloc((void**)&gpuResult, gpuArrSize);
	
	// Copy memory to GPU
	cudaMemcpy(gpuArrayOne, arrayOne, gpuArrSize, cudaMemcpyHostToDevice); 
	
	// Run and time the conditional branching kernel
	clock_t begin = clock();
	gpu_branch<<<numBlocks, blockSize>>>(gpuArrayOne, gpuResult);
	cudaMemcpy(arrayResult, gpuResult, gpuArrSize, cudaMemcpyDeviceToHost); 
	clock_t end = clock();
	
	// Calculate the time required and print
	double elapsed = (double)(end-begin) / CLOCKS_PER_SEC;
	printf("Time elapsed GPU conditional: %fms\n", elapsed*1000);
	
	// Run and time the non-conditional branching kernel
	begin = clock();
	gpu_no_branch<<<numBlocks, blockSize>>>(gpuArrayOne, gpuResult);
	cudaMemcpy(arrayResult, gpuResult, gpuArrSize, cudaMemcpyDeviceToHost); 
	end = clock();
	
	// Calculate the time required and print
	elapsed = (double)(end-begin) / CLOCKS_PER_SEC;
	printf("Time elapsed GPU non-conditional: %fms\n", elapsed*1000);

	// Run and time the conditional cpu code
	begin = clock();
	cpu_branch (arrayOne, arrayResult);
	end = clock();
	
	// Calculate the time required and print
	elapsed = (double)(end-begin) / CLOCKS_PER_SEC;
	printf("Time elapsed CPU conditional: %fms\n", elapsed*1000);
}

// GPU add kernel
// Adds the values at the first two threads into the result thread
__global__
void gpu_add (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = blockOne[thread_idx] + blockTwo[thread_idx];
}

// GPU sub kernel
// Subs the values at the first two threads into the result thread
__global__
void gpu_sub (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = blockOne[thread_idx] - blockTwo[thread_idx];
}

// GPU mult kernel
// Mults the values at the first two threads into the result thread
__global__
void gpu_mult (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = blockOne[thread_idx] * blockTwo[thread_idx];
}

// GPU mod kernel
// Mods the values at the first two threads into the result thread
__global__
void gpu_mod (int32_t* blockOne, int32_t* blockTwo, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = blockOne[thread_idx] % blockTwo[thread_idx];
}

// GPU branch kernel
// Conditionally branches by doing modulo 2 conditional on an
// input array that goes from 0-thread_count and does arbitrary calculations
__global__
void gpu_branch (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (blockOne[thread_idx]%2 == 0) {
		resultBlock[thread_idx] = blockOne[thread_idx]*2;
	}
	else {
		resultBlock[thread_idx] = blockOne[thread_idx]+1;
	}
}

// GPU no branch kernel
// Just assigns the result to the current thread index
__global__
void gpu_no_branch (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	resultBlock[thread_idx] = thread_idx;
}

// CPU branch helper function
// Does the same functionality as the GPU branch kernel on the CPU
void cpu_branch (int32_t* arrayOne, int32_t* resultArray) {
	for (int i=0; i<threadCount; i++) {
		if (arrayOne[i]%2 == 0) {
			resultArray[i] = arrayOne[i]*2;
		}
		else {
			resultArray[i] = arrayOne[i]+1;
		}
	}
}

// Print helper function
// Prints all of the data in the array ordered in blocks
void print_blocks (int32_t* resultArr) {
	for (int i=0; i<numBlocks; i++) {
		printf("B%-2d ", i);
	}
	printf("\n");

	for (int i=0; i<blockSize; i++) {
		for (int x=0; x<numBlocks; x++) {
			printf("%-3d ", resultArr[i + (x*blockSize)]);
		}
		printf("\n");
	}
}
