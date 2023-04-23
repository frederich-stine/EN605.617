# Module 12

This lab modified an OpenCL program that utilized buffers and sub-buffers. I kept the original main buffer size as 16. 
Utilizing a sub-buffer of 2x2 did not make sense to me in this scenario because the input buffer was one dimentional. 
Instead I used a sub-buffer of 4x1. This sub-buffer range was used with a kernel that did a simple average on values. 
The averaging filter simply added all four values and divided by four. I just used this with integers, but could be very 
easily expanded to floating point by changing the inputs. The assignment stated that the output must be 16 bit wide as well. 
The result of a true convolution here would have a size of 13 so the last three values are reading in data that is not correct. 
I did this to meet the requirements of the assignment, but in the real world I would not use this technique.

I used events for profiling all of the calls to the kernel, there were 16 in total for all of the sub-buffers.

There are no arguments passed to this assignment. You can still select the device to run on, but I removed the map functionality. 

I additionally created a *run.sh* script that runs three executions of the program. The input values are fixed so the timing results do not show variance. A log of this *run.sh* script is available in [documentation/full_run.log](documentation/full_run.log).

Here is just the result from one of these runs.
```
Press enter to continue:
OpenCL sub-buffer
Run 1
Simple buffer and sub-buffer Example - FLS
Number of platforms: 	1
	CL_PLATFORM_VENDOR:	NVIDIA Corporation
		CL_DEVICE_TYPE:	CL_DEVICE_TYPE_GPU
 1 2 3 4 5 6 7 8 9 10 11 12 13 10 7 3
Program completed successfully

OpenCl Execution time of kernel 0: 0.010 milliseconds 
OpenCl Execution time of kernel 1: 0.003 milliseconds 
OpenCl Execution time of kernel 2: 0.003 milliseconds 
OpenCl Execution time of kernel 3: 0.003 milliseconds 
OpenCl Execution time of kernel 4: 0.003 milliseconds 
OpenCl Execution time of kernel 5: 0.003 milliseconds 
OpenCl Execution time of kernel 6: 0.002 milliseconds 
OpenCl Execution time of kernel 7: 0.002 milliseconds 
OpenCl Execution time of kernel 8: 0.003 milliseconds 
OpenCl Execution time of kernel 9: 0.003 milliseconds 
OpenCl Execution time of kernel 10: 0.003 milliseconds 
OpenCl Execution time of kernel 11: 0.003 milliseconds 
OpenCl Execution time of kernel 12: 0.003 milliseconds 
OpenCl Execution time of kernel 13: 0.003 milliseconds 
OpenCl Execution time of kernel 14: 0.003 milliseconds 
OpenCl Execution time of kernel 15: 0.003 milliseconds 
Total kernel execution time is: 0.054 milliseconds 
```

Here we can see that the first kernel uses more time than the others. This is probably due to setting up some of the context on the GPU. Overall the execution time of these very small kernels is very short.

This assignment was helpful in learning how to use buffers and sub-buffers in OpenCL. I would not use the actual functionality of this program in the real world, but I would use the knowledge that I learned on buffers and sub-buffers in different use cases.
