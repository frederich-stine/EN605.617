// Frederich Stine EN.605.617
// Module 8 Assignment Part 2

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include <cufft.h>

/******************* Data Type Definitions ********************/
// Structure representing the header of a wav file
typedef struct {
	uint8_t ChunkID[4];
	uint32_t ChunkSize;
	uint8_t Format[4];
	uint8_t SubChunkID[4];
	uint32_t SubChunkSize;
	uint16_t AudioFormat;
	uint16_t NumChannels;
	uint32_t SampleRate;
	uint32_t ByteRate;
	uint16_t BlockAlign;
	uint16_t BitsPerSample;
	uint8_t SubChunk2ID[4];
	uint32_t SubChunk2Size;
} wavData;

/******************* CUDA Kernel Prototypes ********************/

/******************* Core Function Prototypes ********************/
// Function to run an FFT on a simple audio file
void runFFT (void);

/******************* Helper Function Prototypes ********************/

/******************* Global Variables ********************/
int numSamples, resultSize, resultSizeBytes;

/******************* Funtion definitions ********************/
int main (int argc, char** argv) {

	// Prints out a help menu if not enough params are passed
	if (argc != 2) {
		printf("Simple audio spectrum FFT example\n");
		printf("    Call ./assignment {FFT_SIZE} \n");
		exit(0);
	}

	// Set up data sizes
	numSamples = atoi(argv[1]);
	resultSize = (numSamples/2)+1;
	resultSizeBytes = resultSize*sizeof(cufftComplex);

	// Run core function
	runFFT();
}

// Core function that reads from a wav file and runs a single 1d fft
void runFFT (void) {
	FILE* audioFh;
	wavData wavHeader;

	// Open WAV file for processing
	audioFh = fopen("440Hz_44100Hz_16bit_05sec.wav", "rb");
	fread(&wavHeader, 1, sizeof(wavData), audioFh);

	// Print out some data about the file
	printf("Sample rate: %d\n", wavHeader.SampleRate);
	printf("Bits per sample: %d\n", wavHeader.BitsPerSample);
	printf("Num channels: %d\n", wavHeader.NumChannels);

	// Prepare input data buffers
	cufftReal* i_cu_buf;
	cufftComplex* o_cu_buf;
	cudaMallocHost((void**)&i_cu_buf, \
			numSamples*sizeof(cufftReal));
	cudaMallocHost((void**)&o_cu_buf, \
			numSamples*sizeof(cufftComplex));

	// Read in samples from wav file
	int16_t input_buf[numSamples];
	fread(input_buf, numSamples*2, 1, audioFh);

	// Convert wav file to float values
	for(int i=0; i<numSamples; i++) {
		i_cu_buf[i] = (cufftReal)input_buf[i];
		//printf("Input value: %f\n", i_cu_buf[i]);
	}

	// Allocate device data
	cufftReal* d_i_cu_buf;
	cufftComplex* d_o_cu_buf;

	cudaMalloc((void **)&d_i_cu_buf, \
			numSamples*sizeof(cufftReal));
	cudaMalloc((void **)&d_o_cu_buf, resultSizeBytes);

	// Copy input to device
	cudaMemcpy(d_i_cu_buf, i_cu_buf, \
			numSamples*sizeof(cufftReal),\
			cudaMemcpyHostToDevice);

	// Create cufft 1d plan
	// R2C conversion with 1 batch size
	cufftHandle plan;
	cufftPlan1d(&plan, numSamples, CUFFT_R2C, 1);

	// Execute the fft
	cufftExecR2C(plan, d_i_cu_buf, d_o_cu_buf);

	// Copy result back from GPU
	cudaMemcpy(o_cu_buf, d_o_cu_buf, \
			resultSizeBytes, cudaMemcpyDeviceToHost);

	// Calculate magnitude
	float* d_o_magnitude = (float*) malloc (resultSize*sizeof(float));
	for (int i=0; i<resultSize; i++) {
		d_o_magnitude[i] = sqrt(pow(o_cu_buf[i].x, 2) + pow(o_cu_buf[i].y, 2));
	}

	// Calculate magnitude db
	float* d_o_magnitude_db = (float*) malloc (resultSize*sizeof(float));
	for (int i=0; i<resultSize; i++) {
		d_o_magnitude_db[i] = 20*log10(d_o_magnitude[i]);
	}

	// Print out information about the results
	for (int i=0; i<resultSize; i++) {
		float frequency = (float)i*(float)wavHeader.SampleRate/(float)numSamples;
		printf("FFT Result: Frequency: %f: C: %f: I: %f \n"\
				"    Magnitude: %f: Magnitude dB: %f\n", \
				frequency, o_cu_buf[i].x, o_cu_buf[i].y, \
				d_o_magnitude[i], d_o_magnitude_db[i]);
	}

	// Free all memory
	cufftDestroy(plan);

	cudaFree(d_i_cu_buf);
	cudaFree(d_o_cu_buf);

	cudaFreeHost(i_cu_buf);
	cudaFreeHost(o_cu_buf);

	free(d_o_magnitude);
	free(d_o_magnitude_db);
}

