#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include "info.hpp"

#define DEFAULT_PLATFORM 0

float width;
float count;

// Function to check and handle OpenCL errors
inline void 
checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " <<  name << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void readFile(char* fileName, float*** input) {
    
    FILE* fh = fopen(fileName, "r");
    if (fh == NULL) {
        printf("Error: Invalid file");
        exit(0);
    }
    fscanf(fh, "%f\n", &width);
    fscanf(fh, "%f\n", &count);


    *input = (float**)malloc(sizeof(float*)*count);
    for (int i=0; i<int(count); i++) {
        (*input)[i] = (float*) malloc(sizeof(float)*width);
    }

    printf("Count: %f\n", count);
    printf("Width: %f\n", width);

    for (int i=0; i<(int)count; i++) {
        for (int x=0; x<(int)width; x++) {
            fscanf(fh, "%f,", &(*input)[i][x]);
        }
        fseek(fh, 1, SEEK_CUR);
    }

    /*for (int i=0; i<(int)count; i++) {
        printf("%f\n", (*input)[i][0]);
    }*/

    fclose(fh);
}


void freeInput(float*** input) {

    for (int i=0; i<(int)count; i++) {
        free((*input)[i]);
    }

    free(*input);

}


///
//	main() for simple buffer and sub-buffer example
//
int main(int argc, char** argv)
{

    if (argc != 2) {
        printf("Error: Not enough arguments\n");
        printf("Correct usage is:\n");
        printf("    : ./assignment {filename}");
        exit(0);
    }

    float** input;
    readFile(argv[1], &input);

    cl_int errNum;
    cl_uint numPlatforms;
    cl_uint numDevices;
    cl_platform_id * platformIDs;
    cl_device_id * deviceIDs;
    cl_context context;
    cl_program program;

    int platform = DEFAULT_PLATFORM; 

    // First, select an OpenCL platform to run on.  
    errNum = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkErr( 
        (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
        "clGetPlatformIDs"); 
 
    platformIDs = (cl_platform_id *)alloca(
            sizeof(cl_platform_id) * numPlatforms);

    std::cout << "Number of platforms: \t" << numPlatforms << std::endl; 

    errNum = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
    checkErr( 
       (errNum != CL_SUCCESS) ? errNum : (numPlatforms <= 0 ? -1 : CL_SUCCESS), 
       "clGetPlatformIDs");

    std::ifstream srcFile("assignment.cl");
    checkErr(srcFile.is_open() ? CL_SUCCESS : -1, "reading assignment.cl");

    std::string srcProg(
        std::istreambuf_iterator<char>(srcFile),
        (std::istreambuf_iterator<char>()));

    const char * src = srcProg.c_str();
    size_t length = srcProg.length();

    deviceIDs = NULL;
    DisplayPlatformInfo(
        platformIDs[platform], 
        CL_PLATFORM_VENDOR, 
        "CL_PLATFORM_VENDOR");

    errNum = clGetDeviceIDs(
        platformIDs[platform], 
        CL_DEVICE_TYPE_ALL, 
        0,
        NULL,
        &numDevices);
    if (errNum != CL_SUCCESS && errNum != CL_DEVICE_NOT_FOUND)
    {
        checkErr(errNum, "clGetDeviceIDs");
    }       

    deviceIDs = (cl_device_id *)alloca(sizeof(cl_device_id) * numDevices);
    errNum = clGetDeviceIDs(
        platformIDs[platform],
        CL_DEVICE_TYPE_ALL,
        numDevices, 
        &deviceIDs[0], 
        NULL);
    checkErr(errNum, "clGetDeviceIDs");

    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platformIDs[platform],
        0
    };

    context = clCreateContext(
        contextProperties, 
        numDevices,
        deviceIDs, 
        NULL,
        NULL, 
        &errNum);
    checkErr(errNum, "clCreateContext");
 
    // Create program from source
    program = clCreateProgramWithSource(
        context, 
        1, 
        &src, 
        &length, 
        &errNum);
    checkErr(errNum, "clCreateProgramWithSource");
 
    // Build program
    errNum = clBuildProgram(
        program,
        numDevices,
        deviceIDs,
        "-I.",
        NULL,
        NULL);

 
    if (errNum != CL_SUCCESS) 
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(
            program, 
            deviceIDs[0], 
            CL_PROGRAM_BUILD_LOG,
            sizeof(buildLog), 
            buildLog, 
            NULL);

            std::cerr << "Error in OpenCL C source: " << std::endl;
            std::cerr << buildLog;
            checkErr(errNum, "clBuildProgram");
    }

    std::vector<cl_mem> buffers;
    // create a buffer for each row to cover all the input data
    for (int i=0; i<count; i++) {
        cl_mem buffer = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(float)*width,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");

        buffers.push_back(buffer);
    }
 
    cl_mem bufOut = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            sizeof(float)*count,
            NULL,
            &errNum);
        checkErr(errNum, "clCreateBuffer");

    // Create command queues
    InfoDevice<cl_device_type>::display(
     	deviceIDs[0], 
     	CL_DEVICE_TYPE, 
     	"CL_DEVICE_TYPE");

    cl_command_queue queue = 
     	clCreateCommandQueue(
     	context,
     	deviceIDs[0],
     	0,
     	&errNum);
    checkErr(errNum, "clCreateCommandQueue");

    std::vector<cl_kernel> squareKernels;
    for (int i=0; i<count; i++) {
        cl_kernel kernel = clCreateKernel(
        program,
        "square",
        &errNum);
        checkErr(errNum, "clCreateKernel(square)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        checkErr(errNum, "clSetKernelArg(square)");

        squareKernels.push_back(kernel);
    }
     
    std::vector<cl_kernel> averageKernels;
       for (int i=0; i<count; i++) {
        cl_kernel kernel = clCreateKernel(
        program,
        "average",
        &errNum);
        checkErr(errNum, "clCreateKernel(average)");

        errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&buffers[i]);
        errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufOut);
        errNum = clSetKernelArg(kernel, 2, sizeof(cl_float), (void *)&width);
        errNum = clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&i);
        checkErr(errNum, "clSetKernelArg(average)");

        averageKernels.push_back(kernel);
    }
 
    // Write input data
    for (int i=0; i<count; i++) {
        errNum = clEnqueueWriteBuffer(
        queue,
        buffers[i],
        CL_TRUE,
        0,
        sizeof(float) * width,
        (void*)input[i],
        0,
        NULL,
        NULL);
    } 
 
    std::vector<cl_event> events;
    // call kernel for each device
    size_t gWI = (int)width;
	
    std::chrono::time_point startTime = std::chrono::high_resolution_clock::now();

    for (int i=0; i<count; i++) {
        cl_event event;

        errNum = clEnqueueNDRangeKernel(
        queue, 
        squareKernels[i], 
        1, 
        NULL,
        (const size_t*)&gWI, 
        (const size_t*)NULL, 
        0, 
        0, 
        &event);

        events.push_back(event);
    }

    clWaitForEvents(events.size(), &events[0]);
    
	std::chrono::time_point stopTime = std::chrono::high_resolution_clock::now();
	auto ns1 = std::chrono::duration<double>(stopTime - startTime);

    std::vector<cl_event> avgEvents;
    // call kernel for each device
    gWI = 1;

	startTime = std::chrono::high_resolution_clock::now();

    for (int i=0; i<count; i++) {
        cl_event event;

        errNum = clEnqueueNDRangeKernel(
        queue, 
        averageKernels[i], 
        1, 
        NULL,
        (const size_t*)&gWI, 
        (const size_t*)NULL, 
        0, 
        0, 
        &event);

        avgEvents.push_back(event);
    } 

    clWaitForEvents(avgEvents.size(), &avgEvents[0]);

	stopTime = std::chrono::high_resolution_clock::now();
	auto ns2 = std::chrono::duration<double>(stopTime - startTime);

    float* output = (float*) malloc(sizeof(float)*count);

 	// Read back computed data
    clEnqueueReadBuffer(
            queue,
            bufOut,
            CL_TRUE,
            0,
            sizeof(float) * count,
            (void*)output,
            0,
            NULL,
            NULL);


    printf("\nResult after double and average: \n");
    for (int i=0; i<(int)count; i++) {
        printf("Row: %d: %f\n", i, output[i]);
    }

	printf("\nSquare kernel runtime: %lfs\n", ns1.count());
	printf("Average kernel runtime: %lfs\n", ns2.count());

    free(output);
    freeInput(&input);

    return 0;
}
