// Frederich Stine EN.605.617
// Module 4 Assignment

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/******************* CUDA Function Prototypes ********************/
// CUDA kernel for the caesar cipher
__global__
void caesar_cipher (int rot, char* blockOne, char* resultBlock);

/******************* Core Function Prototypes ********************/
// Core function that handles running the caesar cipher kernel
char* run_caesar_cipher(int rot, char* input, int size);

/******************* Helper Function Prototypes ********************/
// This function reads in a file for processing and returns a pointer
// to the file contents in memory
char* read_input_file(char* input_file);
// This function writes the processed contents out to a specified file
void write_output_file(char* output_file, char* output, int size);

// Global variables used throughout file
uint32_t blockSize = 0;

/******************* Funtion definitions ********************/
// Main function
// This function takes in command line arguments and invokes the
// run caesar cipher function
int main(int argc, char** argv) {
	// Prints out a help menu if not enough params are passed
	if (argc != 5) {
		printf("Call ./caesar_cipher {blockSize} {input_file} {output_file} {rot}\n");
		exit(0);
	}

	// Load the parameters from the command line
	blockSize = atoi(argv[1]);
	char* input_file = argv[2];
	char* output_file = argv[3];
	int rot = atoi(argv[4]);
	
	// Process input file
	char* buf = read_input_file(input_file);
	int bufLen = strlen(buf);

	// Call caesar cipher run function
	char* result = run_caesar_cipher(rot, buf, bufLen);

	// Write output to disk
	write_output_file(output_file, result, bufLen);
	
	// IMPORTANT - clean heap memory
	free(result);
	free(buf);
}

// Function to call caesar cipher kernel properly
// This function takes in a buffer of text, the size of the buffer,
// and a value to rotate by
char* run_caesar_cipher(int rot, char* input, int size) {
	// Allocate memory for the result
	char* result = (char*)malloc(size);
	char *d_input, *d_result;

	// Allocate GPU memory for input and result
	cudaMalloc((void**)&d_input, size);
	cudaMalloc((void**)&d_result, size);

	// Copy data to the GPU
	cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);

	// Call caesar cipher kernel
	caesar_cipher<<<((size+blockSize-1)/blockSize),
		blockSize>>>(rot, d_input, d_result);

	// Copy result from GPU to host
	cudaMemcpy(result, d_result, size, cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_input);
	cudaFree(d_result);

	// Return result pointer
	return result;
}

// CUDA Kernel to execute a caesar cipher
__global__
void caesar_cipher (int rot, char* blockOne, char* resultBlock) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	// Check for a capital letter
	if ( blockOne[thread_idx] >= 'A' && blockOne[thread_idx] <= 'Z') {
		// Rotate and shift - modulo for wrap around
		resultBlock[thread_idx] = (((blockOne[thread_idx] - 'A') + rot) % 26) + 'A';
	}
	// Check for lowercase letter
	else if ( blockOne[thread_idx] >= 'a' && blockOne[thread_idx] <= 'z') {
		// Rotate and shift - modulo for wrap around
		resultBlock[thread_idx] = (((blockOne[thread_idx] - 'a') + rot) % 26) + 'a';
	}
	// Do nothing with special characters
	else {
		resultBlock[thread_idx] = blockOne[thread_idx];
	}
}

// Helper function to read an input file
char* read_input_file(char* input_file) {
	FILE* fp;
	int size;
	
	// Open file for reading and check for success
	fp = fopen(input_file, "r");
	if(fp == NULL) {
		printf("Provided input file is invalid\n");
		exit(0);
	}
	// Deterine file length
	fseek(fp, 0, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	
	// Read in whole file to buffer
	char* buffer = (char*)malloc(size);
	fread(buffer, 1, size, fp);

	// Close file and return
	fclose(fp);
	return buffer;
}

//  Helper function to write buffer to an output file
void write_output_file(char* output_file, char* output, int size) {
	FILE* fp;

	// Open file to writing and write contents to file
	fp = fopen(output_file, "w");
	fwrite(output, 1, size, fp);

	// Close file
	fclose(fp);
}
