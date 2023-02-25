# Module 5 - Frederich Stine

## Shared and Constant Memory

To demonstrate the ability to use both shared and constant memory I created four separate kernels.
- One that copies data into shared memory before performing calculations
- One that uses shared memory as local memory inside of the kernel
- One that copies data to a constant array with CudaMemcpyToSymbol
- One that uses some hardcoded constant values

All four of these kernels perform the same operations. The first three perform the same operations on the same quantity of data.
The final one that uses hardcoded values uses 4 pre-set values instead of utilizing an array of 1024 initialized values.

All of the kernels run with a threadCount of 1024. This limit is imposed by the constant functions where
memory has to be statically declared. The shared memory kernels can be run with variable threadCounts by 
allowing this as a user input, but was not done for this assignment as would not provide a valid comparison.

As with previous assignments, this one *assignment.cu* file contains all of the functionality and uses 
command line arguments to choose between different functions. This also allows for the use of a variable 
block size. The help menu for this is below:

```
Call ./assignment {blockSize} {operation}
Operations: 
    0: Copy to shared Memory
    1: Shared memory for local
    2: Copy to constant memory
    3: Constant memory only
```

Using this knowledge we can compare the performance again using *nvprof*. It will be good to note that some 
of the calls require differing amounts of memcpy and allocation. I still enjoy the fine amount of detail that 
nvprof allows over simple timing of single or grouped functions.

I additionally created a *run.sh* script which runs all four kernels with two different block sizes. The output 
of half of this with nvprof is shown below. For sake of brevity I have removed the textural output from the program 
and only left the timing statistics.

```
Frederich Stine - EN 605.617 - JHU EP
----------------------------------------------------------------
Example runner to show execution of Module 5 assignment
----------------------------------------------------------------
This is a provided example runner of how to properly use
my module 5 assignment.
This assignment runs through 4 different kernels:
    0: Arith with full copy to shared memory
    1: Arith with shared memory for local variables
    2: Arith with copy to constant memory
    3: Arith with constant memory only

These assignment runs with a fixed quantity of threads
at 1024. There are two varying block sizes in this runner.

Press enter to continue:Portion 1 - Copy to shared memory
Running with block size of 128 -
==10124== NVPROF is profiling process 10124, command: ./assignment 128 0
==10124== Profiling application: ./assignment 128 0
==10124== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.99%  2.6880us         1  2.6880us  2.6880us  2.6880us  gpu_all_arith_shared_copy(int*, int*)
                   29.73%  1.6650us         1  1.6650us  1.6650us  1.6650us  [CUDA memcpy HtoD]
                   22.28%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   97.67%  130.37ms         2  65.186ms  4.7290us  130.37ms  cudaMallocHost
                    0.87%  1.1627ms         1  1.1627ms  1.1627ms  1.1627ms  cuLibraryLoadData
                    0.50%  669.69us         1  669.69us  669.69us  669.69us  cuDeviceGetName
                    0.24%  326.73us         2  163.36us  4.9090us  321.82us  cudaFreeHost
                    0.24%  326.46us         2  163.23us  15.679us  310.79us  cudaMemcpy
                    0.23%  308.94us         1  308.94us  308.94us  308.94us  cuDeviceGetPCIBusId
                    0.09%  119.33us       114  1.0460us     110ns  50.345us  cuDeviceGetAttribute
                    0.07%  89.318us         2  44.659us  7.1230us  82.195us  cudaFree
                    0.06%  73.599us         2  36.799us  2.8550us  70.744us  cudaMalloc
                    0.02%  24.817us         1  24.817us  24.817us  24.817us  cudaLaunchKernel
                    0.00%  1.7040us         3     568ns     200ns  1.1830us  cuDeviceGetCount
                    0.00%     672ns         2     336ns     151ns     521ns  cuDeviceGet
                    0.00%     301ns         1     301ns     301ns     301ns  cuModuleGetLoadingMode
                    0.00%     301ns         1     301ns     301ns     301ns  cuDeviceTotalMem
                    0.00%     161ns         1     161ns     161ns     161ns  cuDeviceGetUuid
Press enter to continue:Portion 2 - Shared memory for local
Running with block size of 128 -
==10139== NVPROF is profiling process 10139, command: ./assignment 128 1
==10139== Profiling application: ./assignment 128 1
==10139== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   46.78%  2.5600us         1  2.5600us  2.5600us  2.5600us  gpu_all_arith_shared(int*, int*)
                   30.41%  1.6640us         1  1.6640us  1.6640us  1.6640us  [CUDA memcpy HtoD]
                   22.81%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   99.02%  76.503ms         2  38.252ms  2.8350us  76.501ms  cudaMallocHost
                    0.39%  300.83us         2  150.41us  3.5670us  297.26us  cudaFreeHost
                    0.15%  115.12us         1  115.12us  115.12us  115.12us  cuLibraryLoadData
                    0.14%  107.13us       114     939ns     120ns  46.287us  cuDeviceGetAttribute
                    0.10%  76.755us         2  38.377us  5.4700us  71.285us  cudaFree
                    0.07%  54.932us         2  27.466us  1.9630us  52.969us  cudaMalloc
                    0.05%  35.848us         1  35.848us  35.848us  35.848us  cuDeviceGetName
                    0.04%  34.075us         2  17.037us  14.748us  19.327us  cudaMemcpy
                    0.03%  20.729us         1  20.729us  20.729us  20.729us  cudaLaunchKernel
                    0.01%  5.2200us         1  5.2200us  5.2200us  5.2200us  cuDeviceGetPCIBusId
                    0.00%  1.4130us         3     471ns     190ns     962ns  cuDeviceGetCount
                    0.00%     620ns         2     310ns     150ns     470ns  cuDeviceGet
                    0.00%     311ns         1     311ns     311ns     311ns  cuModuleGetLoadingMode
                    0.00%     270ns         1     270ns     270ns     270ns  cuDeviceTotalMem
                    0.00%     190ns         1     190ns     190ns     190ns  cuDeviceGetUuid
Press enter to continue:Portion 3 - Copy to constant memory
Running with block size of 128 -
==10153== NVPROF is profiling process 10153, command: ./assignment 128 2
==10153== Profiling application: ./assignment 128 2
==10153== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   47.43%  2.6560us         1  2.6560us  2.6560us  2.6560us  gpu_all_arith_const(int*)
                   30.29%  1.6960us         1  1.6960us  1.6960us  1.6960us  [CUDA memcpy HtoD]
                   22.29%  1.2480us         1  1.2480us  1.2480us  1.2480us  [CUDA memcpy DtoH]
      API calls:   99.05%  75.764ms         1  75.764ms  75.764ms  75.764ms  cudaMallocHost
                    0.39%  296.43us         1  296.43us  296.43us  296.43us  cudaFreeHost
                    0.16%  120.19us         1  120.19us  120.19us  120.19us  cuLibraryLoadData
                    0.14%  104.46us       114     916ns     110ns  45.516us  cuDeviceGetAttribute
                    0.09%  72.367us         1  72.367us  72.367us  72.367us  cudaFree
                    0.07%  53.931us         1  53.931us  53.931us  53.931us  cudaMalloc
                    0.03%  20.518us         1  20.518us  20.518us  20.518us  cudaLaunchKernel
                    0.03%  19.296us         1  19.296us  19.296us  19.296us  cuDeviceGetName
                    0.02%  14.477us         1  14.477us  14.477us  14.477us  cudaMemcpy
                    0.02%  13.615us         1  13.615us  13.615us  13.615us  cudaMemcpyToSymbol
                    0.01%  9.0770us         1  9.0770us  9.0770us  9.0770us  cuDeviceGetPCIBusId
                    0.00%  1.1830us         3     394ns     171ns     752ns  cuDeviceGetCount
                    0.00%     661ns         2     330ns     130ns     531ns  cuDeviceGet
                    0.00%     311ns         1     311ns     311ns     311ns  cuDeviceTotalMem
                    0.00%     220ns         1     220ns     220ns     220ns  cuModuleGetLoadingMode
                    0.00%     170ns         1     170ns     170ns     170ns  cuDeviceGetUuid
Press enter to continue:Portion 4 - Constant memory only
Running with block size of 128 -
==10167== NVPROF is profiling process 10167, command: ./assignment 128 3
==10167== Profiling application: ./assignment 128 3
==10167== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   67.21%  2.6240us         1  2.6240us  2.6240us  2.6240us  gpu_all_arith_only_const(int*)
                   32.79%  1.2800us         1  1.2800us  1.2800us  1.2800us  [CUDA memcpy DtoH]
      API calls:   99.04%  75.770ms         1  75.770ms  75.770ms  75.770ms  cudaMallocHost
                    0.42%  318.09us         1  318.09us  318.09us  318.09us  cudaFreeHost
                    0.16%  118.99us         1  118.99us  118.99us  118.99us  cuLibraryLoadData
                    0.14%  104.71us       114     918ns     110ns  45.135us  cuDeviceGetAttribute
                    0.10%  73.769us         1  73.769us  73.769us  73.769us  cudaFree
                    0.07%  51.918us         1  51.918us  51.918us  51.918us  cudaMalloc
                    0.03%  24.035us         1  24.035us  24.035us  24.035us  cudaLaunchKernel
                    0.03%  20.508us         1  20.508us  20.508us  20.508us  cuDeviceGetName
                    0.02%  18.334us         1  18.334us  18.334us  18.334us  cudaMemcpy
                    0.01%  4.8990us         1  4.8990us  4.8990us  4.8990us  cuDeviceGetPCIBusId
                    0.00%  1.5120us         3     504ns     170ns  1.1020us  cuDeviceGetCount
                    0.00%     571ns         2     285ns     130ns     441ns  cuDeviceGet
                    0.00%     400ns         1     400ns     400ns     400ns  cuDeviceTotalMem
                    0.00%     270ns         1     270ns     270ns     270ns  cuModuleGetLoadingMode
                    0.00%     161ns         1     161ns     161ns     161ns  cuDeviceGetUuid
```

From these runs we can compare the different in runtime of the kernels with the different amounts of time that it took to allocate and copy memory. 
Interestingly the kernel execute time of all four of these kernels are within a margin of error of each other. If the kernelsize was much larger 
we may be able to see a different in the actual execution time, but the effects are minor. Performing more intensive calculations inside of the kernel 
besides the four simple arithmetic operations would also likely lead to a faster execution time.

Interestingly the kernel was just as fast when copying everyting into shared memory vs only using shared memory for local variables. This shows that 
local memory is very fast as the additional overhead did not result in any sizeable performance deduction or gain. 

As expected the constant memory portion required less data to be copied back and forth from global to device memory. This resulted in less time being 
spent copying memory. This is easily seen in the lack of a copy to device entirely in the last capture, but this also provides less functionality than
both the shared and copy to global examples.

After completing this assignment I understand the use cases of both shared and constant memory. I think that these kernels forced this functionality, 
but when programming in Cuda naturally the use for shared and global variables would become very clear.

## Stretch Problem

