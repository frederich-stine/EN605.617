# Module 6 - Frederich Stine

## Registers

To demonstrate the speed difference of registers versus global memory I created three separate kernels

- One that copies all data used into registers
- One that uses registers as a local variable
- One that purely uses global memory

These kernels, differently from the previous assignment, perform operations in a loop of 10000 iterations inside of the kernel.
With the last assignment I found that it was hard to see the real difference between memory speedup due to how fast 
the kernels executed. This made the relationship very easy to see, but I think thta the previous assignment is 
still a valid testing case as the launch and copy time of the kernel will likely always be a large portion of the 
execution time.

This program allows for a variable size of threads and block size. The most register intensive kernel only utilizes 4 
registers, so a lack of registers on the device should not be an issue.

As with previous assignments, all functionality is packed into one *assignment.cu* file. Different functions are invoked 
from the command line arguments. The help menu is below:

```
Call ./assignment {threadCount} {blockSize} {operation}
Operations: 
    0: Copy to register
    1: Register local variables
    2: Global memory only
```

As per usual I created a *run.sh* script that uses nvprof to time different elements of the program. The output of some 
of the execution of this script is shown below. The full log is available in [documentation/full_run.log](documentation/full_run.log).

```
Frederich Stine - EN 605.617 - JHU EP
----------------------------------------------------------------
Example runner to show execution of Module 6 assignment
----------------------------------------------------------------
This is a provided example runner of how to properly use
my module 6 assignment.
This assignment runs through 3 different kernels:
    0: Copy to registers
    1: Registers for local variables
    2: Global memory only

Press enter to continue:Portion 1 - Copy to registers
Running 1024 threads with block size of 128 -
==27932== NVPROF is profiling process 27932, command: ./assignment 1024 128 0
==27932== Profiling application: ./assignment 1024 128 0
==27932== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.53%  115.14us         1  115.14us  115.14us  115.14us  gpu_register_copy_arith(int*, int*)
                    1.41%  1.6640us         1  1.6640us  1.6640us  1.6640us  [CUDA memcpy HtoD]
                    1.06%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   99.03%  108.68ms         2  54.341ms  2.9250us  108.68ms  cudaMallocHost
                    0.28%  305.34us         2  152.67us  3.4860us  301.85us  cudaFreeHost
                    0.21%  231.63us         1  231.63us  231.63us  231.63us  cuLibraryLoadData
                    0.16%  180.02us       114  1.5790us     240ns  72.697us  cuDeviceGetAttribute
                    0.14%  149.83us         2  74.916us  20.639us  129.19us  cudaMemcpy
                    0.07%  72.897us         2  36.448us  4.4280us  68.469us  cudaFree
                    0.05%  58.660us         2  29.330us  2.1440us  56.516us  cudaMalloc
                    0.03%  30.026us         1  30.026us  30.026us  30.026us  cuDeviceGetName
                    0.02%  22.643us         1  22.643us  22.643us  22.643us  cudaLaunchKernel
                    0.01%  8.2650us         1  8.2650us  8.2650us  8.2650us  cuDeviceGetPCIBusId
                    0.00%  2.3250us         3     775ns     351ns  1.2930us  cuDeviceGetCount
                    0.00%  1.2230us         2     611ns     261ns     962ns  cuDeviceGet
                    0.00%     481ns         1     481ns     481ns     481ns  cuDeviceTotalMem
                    0.00%     450ns         1     450ns     450ns     450ns  cuModuleGetLoadingMode
                    0.00%     341ns         1     341ns     341ns     341ns  cuDeviceGetUuid
Press enter to continue:Portion 2 - Register and global
Running 1024 threads with block size of 128 -
==27946== NVPROF is profiling process 27946, command: ./assignment 1024 128 1
==27946== Profiling application: ./assignment 1024 128 1
==27946== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.55%  656.58us         1  656.58us  656.58us  656.58us  gpu_register_arith(int*, int*)
                    0.26%  1.6960us         1  1.6960us  1.6960us  1.6960us  [CUDA memcpy HtoD]
                    0.19%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   98.38%  98.806ms         2  49.403ms  2.7960us  98.803ms  cudaMallocHost
                    0.69%  689.57us         2  344.79us  19.467us  670.10us  cudaMemcpy
                    0.32%  323.83us         2  161.92us  3.5460us  320.28us  cudaFreeHost
                    0.24%  238.60us         1  238.60us  238.60us  238.60us  cuLibraryLoadData
                    0.17%  170.71us       114  1.4970us     240ns  68.749us  cuDeviceGetAttribute
                    0.07%  72.527us         2  36.263us  4.6990us  67.828us  cudaFree
                    0.05%  54.533us         2  27.266us  1.9140us  52.619us  cudaMalloc
                    0.05%  45.205us         1  45.205us  45.205us  45.205us  cuDeviceGetName
                    0.02%  21.270us         1  21.270us  21.270us  21.270us  cudaLaunchKernel
                    0.01%  5.8110us         1  5.8110us  5.8110us  5.8110us  cuDeviceGetPCIBusId
                    0.00%  2.7050us         3     901ns     371ns  1.7530us  cuDeviceGetCount
                    0.00%     932ns         2     466ns     270ns     662ns  cuDeviceGet
                    0.00%     471ns         1     471ns     471ns     471ns  cuDeviceTotalMem
                    0.00%     461ns         1     461ns     461ns     461ns  cuModuleGetLoadingMode
                    0.00%     340ns         1     340ns     340ns     340ns  cuDeviceGetUuid
Press enter to continue:Portion 3 - Global memory only
Running 1024 threads with block size of 128 -
==27960== NVPROF is profiling process 27960, command: ./assignment 1024 128 2
==27960== Profiling application: ./assignment 1024 128 2
==27960== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.96%  6.6652ms         1  6.6652ms  6.6652ms  6.6652ms  gpu_global_arith(int*, int*)
                    0.03%  1.6970us         1  1.6970us  1.6970us  1.6970us  [CUDA memcpy HtoD]
                    0.02%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   92.03%  87.455ms         2  43.727ms  2.9350us  87.452ms  cudaMallocHost
                    7.08%  6.7320ms         2  3.3660ms  55.465us  6.6765ms  cudaMemcpy
                    0.33%  310.30us         2  155.15us  3.5870us  306.71us  cudaFreeHost
                    0.19%  181.14us         1  181.14us  181.14us  181.14us  cuLibraryLoadData
                    0.15%  140.36us       114  1.2310us     180ns  58.019us  cuDeviceGetAttribute
                    0.08%  75.253us         2  37.626us  5.3810us  69.872us  cudaFree
                    0.06%  55.495us         2  27.747us  2.4750us  53.020us  cudaMalloc
                    0.05%  43.582us         1  43.582us  43.582us  43.582us  cuDeviceGetName
                    0.02%  21.440us         1  21.440us  21.440us  21.440us  cudaLaunchKernel
                    0.01%  8.3860us         1  8.3860us  8.3860us  8.3860us  cuDeviceGetPCIBusId
                    0.00%  2.5460us         3     848ns     311ns  1.8340us  cuDeviceGetCount
                    0.00%     851ns         2     425ns     220ns     631ns  cuDeviceGet
                    0.00%     601ns         1     601ns     601ns     601ns  cuModuleGetLoadingMode
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     280ns         1     280ns     280ns     280ns  cuDeviceGetUuid
```

Due to the high amount of loop iterations in the kernel it is very easy to see the change in performance across 
these kernels. The copy to register function is 57.88x faster than the global memory only. Additionally, 
using registers for the writable memory causes a large speedup only slowing down by around 4x. This pattern is 
seen throughout the entire run log.

After completing this assignment I understand the importance of registers on the GPU in relationship to registers on the CPU. 
It is crazy how much high speed memory is available on the GPU vs a traditional CPU environment.

## Stretch Problem

For this stretch problem I am performing a code review of the code provided.

There are lots of good elements of this code:
- The timing is done properly and includes copy memory to and front the device
- I like that this is doing a more complex mathematical operation (matrix multiplication). This is something that I may start incorporation into my assignments in the future.
- This compares two different types of memory (shared vs regs)

There are also some areas for improvement in this code:
- The largest improvement I can think of here is how the registers are used in both the tests and the matrix multiplication function. These registers seem to be used in place of constant memory. The register values are initialized and then are never used\changed. Most all of the computation here is still done in global memory for both examples. A more efficent use of registers would have been to use them to hold the input values for multiplication 
 And copy the values over at the end. The constant values could be held in constant memory with very little slowdown (possibly 0 as the values would not need to be initialized on every kernel). 
 Due to this use of global memory I doubt that this program would show much of a speedup between the two calls.
 - This program utilizes fixed sizes for the matrix multiplication. It would be interesting to see this done with variable sized matrices. 

Overall this program does a decent job at comparing register and shared memory, but the use cases could be improved 
to more naturally show this and see larger performance gains.

