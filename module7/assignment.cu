// Frederich Stine EN.605.617
// Module 7 Assignment

#include <stdio.h>
#include <stdint.h>

/******************* CUDA Kernel Prototypes ********************/
// CUDA Kernel that applies 4 arithmetic functions in a loop on registers
__global__
void gpu_arith (int32_t blockOne, int32_t* resultBlock);

/******************* Core Function Prototypes ********************/
// Core function to run kernel with async streams
void run_stream_arith (void);
// Core function to run kernel with single default stream
void run_synchronous_arith (void);

/******************* Helper Function Prototypes ********************/
// Function to print out blocks easily
void print_blocks (int32_t* resultArr);

/******************* Global Variables ********************/
uint32_t threadCount = 0;
uint32_t blockSize = 0;
uint32_t numBlocks = 0;
uint32_t arrSizeBytes = 0;
uint32_t numStreams = 2;

/******************* Funtion definitions ********************/
int main(int argc, char** argv) {

	// Prints out a help menu if not enough params are passed
	if (argc < 4) {
		printf("Call ./assignment {threadCount} {blockSize} {operation} {numStreams for op 0}\n");
		printf("Operations: \n");
		printf("    0: Stream example\n");
		printf("    1: Synchronous example\n");
		exit(0);
	}

	// Load the parameters from the command line
	threadCount = atoi(argv[1]);
	blockSize = atoi(argv[2]);
	numBlocks = (threadCount+(blockSize-1))/blockSize;
	arrSizeBytes = threadCount*sizeof(int32_t);
	int operation = atoi(argv[3]);
	// Handle fifth param for 0
	if (operation == 0) {
		if (argc != 5) {
			printf("Error: Num streams not provided!\r\n");
			exit(0);
		}
		numStreams = atoi(argv[4]);
		if (numStreams == 0) {
			printf("Error: Num streams cannot be zero!\r\n");
			exit(0);
		}
		if (numBlocks%numStreams != 0) {
			printf("Error: Num blocks must be divisble by num streams!\r\n");
			exit(0);;
		}
	}

	switch (operation) {
	case 0:
		// Run stream example
		run_stream_arith();
		break;
	case 1:
		// Run synchronous example
		run_synchronous_arith();
		break;
	default:
		// Error for incorrect operation
		printf("Error: Incorrect operation encountered!\r\n");
		break;
	}
}

// Kernel for arith with copying to registers
__global__
void gpu_arith (int32_t* blockOne, int32_t* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	// One register for temp value
	int32_t localRegResult;
	localRegResult = 0;

	// Perform calculations on register with global memory
	for (int i=0; i<100; i++) {
		localRegResult += blockOne[thread_idx];
		localRegResult *= blockOne[thread_idx];
		localRegResult -= blockOne[thread_idx];
		localRegResult /= blockOne[thread_idx];
	}

	// Copy back to global mem
	resultBlock[thread_idx] = localRegResult;
}

// Core function that runs kernel with async streams
void run_stream_arith (void) {
	int32_t *one, *result;
	int32_t *d_one, *d_result;

	// Allocated page locked  memory using standard c malloc function
	cudaMallocHost((void**)&one, arrSizeBytes);
	cudaMallocHost((void**)&result, arrSizeBytes);

	// Initialize input
	for(int i=0; i<threadCount; i++) {
		one[i] = i;
	}
	
	// Allocate memory on the GPU for computation
	cudaMalloc((void**)&d_one, arrSizeBytes);
	cudaMalloc((void**)&d_result, arrSizeBytes);

	// Allocate cuda streams
	cudaStream_t stream[numStreams];

	// Initialize cuda streams as non blocking
	for (int i=0; i<numStreams; i++) {
		cudaStreamCreateWithFlags(&stream[i], cudaStreamNonBlocking);
	}

	// Timing functionality 
	cudaEvent_t start, stop;
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 

	// Determine some values specific to streams
	uint32_t blocksPerStream = numBlocks/numStreams;
	uint32_t indexPerStream = threadCount/numStreams;
	uint32_t bytesPerStream = arrSizeBytes/numStreams;

	// Start timing
	cudaEventRecord(start);

	// Run all async memcpys and kernels
	for (int i=0; i<numStreams; i++) {
		int index = i*indexPerStream;
		
		cudaMemcpyAsync(&d_one[index], &one[index], bytesPerStream, cudaMemcpyHostToDevice, stream[i]);
		gpu_arith <<<blocksPerStream, blockSize, 0, stream[i]>>> (&d_one[index], &d_result[index]);
		cudaMemcpyAsync(&result[index], &d_result[index], bytesPerStream, cudaMemcpyDeviceToHost, stream[i]);
	}

	// Stop timing and synchronize
	cudaDeviceSynchronize();
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); 
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 

	printf("Time taken: %3.4f ms\r\n", elapsedTime);

	// Destroy streams
	for (int i=0; i<numStreams; i++) {
		cudaStreamDestroy(stream[i]);
	}

	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_result);

	// Free pinned memory on host
	cudaFreeHost(one);
	cudaFreeHost(result);
}

// Simple core function to call using synchronous stream
void run_synchronous_arith (void) {
	int32_t *one, *result;
	int32_t *d_one, *d_result;

	// Allocated page locked  memory using standard c malloc function
	cudaMallocHost((void**)&one, arrSizeBytes);
	cudaMallocHost((void**)&result, arrSizeBytes);

	// Initialize input
	for(int i=0; i<threadCount; i++) {
		one[i] = i;
	}
	
	// Allocate memory on the GPU for computation
	cudaMalloc((void**)&d_one, arrSizeBytes);
	cudaMalloc((void**)&d_result, arrSizeBytes);

	// Create variables for timing
	cudaEvent_t start, stop;
	cudaEventCreate( &start ); 
	cudaEventCreate( &stop ); 

	// Start timing
	cudaEventRecord(start);

	// Copy and run kernel
	cudaMemcpy(d_one, one, arrSizeBytes, cudaMemcpyHostToDevice);
	
	gpu_arith <<<numBlocks, blockSize>>>(d_one, d_result);

	cudaMemcpy(result, d_result, arrSizeBytes, cudaMemcpyDeviceToHost);

	// Stop timing
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); 
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop); 

	printf("Time taken: %3.4f ms\r\n", elapsedTime);
	
	// Free memory on GPU
	cudaFree(d_one);
	cudaFree(d_result);

	// Free pinned memory on host
	cudaFreeHost(one);
	cudaFreeHost(result);
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
