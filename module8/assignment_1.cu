// Frederich Stine EN.605.617
// Module 8 Assignment Part 1

#include <stdio.h>
#include <stdint.h>
#include <time.h>

#include <curand.h>
#include <curand_kernel.h>

/******************* CUDA Kernel Prototypes ********************/

/******************* Core Function Prototypes ********************/
// Function that generates a variable amount of random numbers with curand
void curand_core (void);

/******************* Helper Function Prototypes ********************/

/******************* Global Variables ********************/
uint32_t threadCount = 0;
uint32_t arrSizeBytes = 0;

/******************* Funtion definitions ********************/
int main (int argc, char** argv) {

	// Prints out a help menu if not enough params are passed
	if (argc != 2) {
		printf("Simple cuRand example host calls\n");
		printf("    Call ./assignment {threadCount}\n");
		exit(0);
	}

	// Load the parameters from the command line
	threadCount = atoi(argv[1]);
	arrSizeBytes = threadCount*sizeof(float);

	// Run core function
	curand_core();
}

// Core function that generates randon numbers
void curand_core (void) {
	// Create generator and host variables
	curandGenerator_t generator;
	float *hostArr, *devArr;

	// Allocate data for random numbers
	cudaMallocHost((void**)&hostArr, arrSizeBytes);
	cudaMalloc((void**)&devArr, arrSizeBytes);

	// Create generator
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
	// Initialize seed with time
	curandSetPseudoRandomGeneratorSeed(generator, time(NULL));
	// Generate numbers
	curandGenerateUniform(generator, devArr, threadCount);

	// Copy numbers back to the host
	cudaMemcpy(hostArr, devArr, arrSizeBytes,\
        cudaMemcpyDeviceToHost);

	// Print the random numbers out
	for (int i=0; i<threadCount; i++) {
		printf("Random %d: %f\n", i, hostArr[i]);
	}

	// Free variables
	curandDestroyGenerator(generator);

	cudaFreeHost(hostArr);
	cudaFree(devArr);
}

