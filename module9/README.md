# Module 9

## Thrust

This lab utilized the CUDA Thrust libraries that abstract kernel calls to use of C++ vector objects.
I used this library to implement the same functionality as we did in module 3: add, sub, mult, and mod.
This assignment is in *thrust_assignment.cu*.

Since thrust is abstracted from the user I did not have the same granularity with thread count and block size.
I compared the timing of this code with my module 3 assignment.

The arguments passed to the thrust_assignment are:
```
Call ./thrust_assignment {vector_size} {operation}}
Operations: 
    0: Add
    1: Sub
    2: Mult
    3: Mod
```

I additionally created a *run.sh* script that runs both the module 3 assignment and the thrust assignment two times.
The output of this file is very long and is available in [documentation/thrust_run.log](documentation/thrust_run.log).

Here is some shortened output from this file when comparing the two, but the full file can be checked for functional requirements.
```
Press enter to continue:
Thrust addition: running with vector size of 10
==15137== NVPROF is profiling process 15137, command: ./thrust_assignment 10 0
Array 1[0]:    0 :Array 2[0]:   10 :Result[0]:   10
Array 1[1]:    1 :Array 2[1]:    9 :Result[1]:   10
Array 1[2]:    2 :Array 2[2]:    8 :Result[2]:   10
Array 1[3]:    3 :Array 2[3]:    7 :Result[3]:   10
Array 1[4]:    4 :Array 2[4]:    6 :Result[4]:   10
Array 1[5]:    5 :Array 2[5]:    5 :Result[5]:   10
Array 1[6]:    6 :Array 2[6]:    4 :Result[6]:   10
Array 1[7]:    7 :Array 2[7]:    3 :Result[7]:   10
Array 1[8]:    8 :Array 2[8]:    2 :Result[8]:   10
Array 1[9]:    9 :Array 2[9]:    1 :Result[9]:   10
==15137== Profiling application: ./thrust_assignment 10 0
==15137== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   84.78%  29.406us        30     980ns     927ns  1.2150us  [CUDA memcpy DtoH]
                    6.46%  2.2400us         1  2.2400us  2.2400us  2.2400us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<int>, int>, unsigned long>, thrust::cuda_cub::__uninitialized_fill::functor<thrust::device_ptr<int>, int>, unsigned long>(thrust::device_ptr<int>, int)
                    3.60%  1.2480us         1  1.2480us  1.2480us  1.2480us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__transform::binary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::cuda_cub::__transform::no_stencil_tag, thrust::plus<int>, thrust::cuda_cub::__transform::always_true_predicate>, long>, thrust::cuda_cub::__transform::binary_transform_f<thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::detail::normal_iterator<thrust::device_ptr<int>>, thrust::cuda_cub::__transform::no_stencil_tag, thrust::plus<int>, thrust::cuda_cub::__transform::always_true_predicate>, long>(thrust::device_ptr<int>, thrust::detail::normal_iterator<thrust::device_ptr<int>>)
                    3.51%  1.2160us         1  1.2160us  1.2160us  1.2160us  void thrust::cuda_cub::core::_kernel_agent<thrust::cuda_cub::__parallel_for::ParallelForAgent<thrust::cuda_cub::__fill::functor<thrust::detail::normal_iterator<thrust::device_ptr<int>>, int>, long>, thrust::cuda_cub::__fill::functor<thrust::detail::normal_iterator<thrust::device_ptr<int>>, int>, long>(thrust::device_ptr<int>, thrust::detail::normal_iterator<thrust::device_ptr<int>>)
                    1.66%     575ns         2     287ns     223ns     352ns  [CUDA memcpy HtoD]
      API calls:   98.94%  97.817ms         3  32.606ms  2.4950us  97.812ms  cudaMalloc
                    0.42%  412.78us         1  412.78us  412.78us  412.78us  cuLibraryLoadData
                    0.24%  238.18us        32  7.4430us  3.0860us  15.359us  cudaMemcpyAsync
                    0.19%  191.32us       114  1.6780us     280ns  73.438us  cuDeviceGetAttribute
                    0.08%  75.112us         3  25.037us  2.2340us  68.199us  cudaFree
                    0.04%  40.162us        35  1.1470us     691ns  7.1730us  cudaStreamSynchronize
                    0.03%  34.215us         1  34.215us  34.215us  34.215us  cuDeviceGetName
                    0.03%  26.029us         3  8.6760us  3.5370us  18.244us  cudaLaunchKernel
                    0.01%  8.8860us         1  8.8860us  8.8860us  8.8860us  cuDeviceGetPCIBusId
                    0.01%  8.1760us         1  8.1760us  8.1760us  8.1760us  cudaFuncGetAttributes
                    0.01%  7.5310us        62     121ns     110ns     321ns  cudaGetLastError
                    0.00%  2.9060us         3     968ns     421ns  1.8740us  cuDeviceGetCount
                    0.00%  2.3150us         7     330ns     271ns     481ns  cudaGetDevice
                    0.00%  1.6340us         3     544ns     281ns  1.0420us  cudaDeviceGetAttribute
                    0.00%     902ns         2     451ns     321ns     581ns  cuDeviceGet
                    0.00%     720ns         6     120ns     100ns     160ns  cudaPeekAtLastError
                    0.00%     691ns         1     691ns     691ns     691ns  cuDeviceTotalMem
                    0.00%     641ns         1     641ns     641ns     641ns  cuModuleGetLoadingMode
                    0.00%     471ns         1     471ns     471ns     471ns  cuDeviceGetUuid
                    0.00%     150ns         1     150ns     150ns     150ns  cudaGetDeviceCount
...
Press enter to continue:
Module 3 subtraction: running with 10 threads - block size of 10
==15266== NVPROF is profiling process 15266, command: ./module3_assignment 10 10 1
Result: 
B0  
-1  
-2  
0   
0   
3   
3   
5   
7   
7   
6   
==15266== Profiling application: ./module3_assignment 10 10 1
==15266== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.25%  2.5280us         1  2.5280us  2.5280us  2.5280us  gpu_sub(int*, int*, int*)
                   29.71%  1.3120us         1  1.3120us  1.3120us  1.3120us  [CUDA memcpy DtoH]
                   13.04%     576ns         2     288ns     224ns     352ns  [CUDA memcpy HtoD]
      API calls:   99.58%  73.612ms         3  24.537ms  2.3040us  73.607ms  cudaMalloc
                    0.16%  115.27us         1  115.27us  115.27us  115.27us  cuLibraryLoadData
                    0.15%  107.40us       114     942ns     120ns  46.268us  cuDeviceGetAttribute
                    0.04%  30.797us         3  10.265us  4.1070us  14.167us  cudaMemcpy
                    0.03%  21.541us         1  21.541us  21.541us  21.541us  cudaLaunchKernel
                    0.03%  20.759us         1  20.759us  20.759us  20.759us  cuDeviceGetName
                    0.01%  9.3780us         1  9.3780us  9.3780us  9.3780us  cuDeviceGetPCIBusId
                    0.00%  1.8030us         3     601ns     180ns  1.3830us  cuDeviceGetCount
                    0.00%     431ns         2     215ns     121ns     310ns  cuDeviceGet
                    0.00%     291ns         1     291ns     291ns     291ns  cuDeviceTotalMem
                    0.00%     270ns         1     270ns     270ns     270ns  cuModuleGetLoadingMode
                    0.00%     191ns         1     191ns     191ns     191ns  cuDeviceGetUuid
```

If we just look at the sum of the GPU activities as a timing metric here - the thrust library makes lots of different calls
to implement the same functionality as my direct implementation. The thrust assignment for this very small data size 
spends around 35ns on the GPU while my iteration from module 3 spends around 5. From looking through the rest of the 
log it seems like the thrust library scales very well and sees only minor slowdowns (1-2) ns when changing to 
a vector size of 1000 while my code sees a slowdown of around 2x. This still puts the non-thrust example well faster 
than the thrust example. Almost all of the calls are memory copies from device to host which I think is due 
to how the access is provided when printing out the results. Since I copy less data in my example this would result 
for some of the slowdown but is still proportionally larger than my module 3 implementation would have been.

I think that the thrust library is a very cool feature for c++ devs to use cuda functionality without having to fully 
understand or write cuda device code. It seems that it is implemented in a robust manner that is scalable but is likely 
slightly slower in specific scenarios due to overhead from having a more generalized approach.

Overall I think that this is a really cool library due to the abstraction and may use it in the future if a quick set-up 
cuda program is needed!
