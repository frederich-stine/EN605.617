// Frederich Stine EN.605.617
// Module 9 NPP Assignment

#include <stdio.h>

#include <FreeImage.h>

#include <npp.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <Exceptions.h>

// Main function
// This function uses npp to perform a Gaussian blur
// on a 8 bit greyscale image
int main (int argc, char** argv) {

	// Prints out a help menu if not enough params are passed
	if (argc != 3) {
		printf("Call ./npp_assignment {input_file} {output_file}}\n");
		exit(0);
	}

	// Load file names
	char* inputFileName = argv[1];
	char* outputFileName = argv[2];

	// Load in image
	printf("Perf: Loading image\n");
	npp::ImageCPU_8u_C1 iHost;
	npp::loadImage(inputFileName, iHost);

	// Allocate image on device 
	npp::ImageNPP_8u_C1 iDeviceSrc(iHost);

	// Set image size
	NppiSize imgSize = {(int)iDeviceSrc.width(), (int)iDeviceSrc.height()};
	// Create destination image on device
	npp::ImageNPP_8u_C1 iDeviceDst(imgSize.width, imgSize.height);

	// Run a gaussian blur on the image
	printf("Perf: Running gaussian filter\n");
	nppiFilterGauss_8u_C1R (iDeviceSrc.data(), iDeviceSrc.pitch(),
			iDeviceDst.data(), iDeviceDst.pitch(), imgSize, NPP_MASK_SIZE_5_X_5);

	// Output the image
	printf("Perf: Saving image\n");
	saveImage(outputFileName, iDeviceDst);
}
