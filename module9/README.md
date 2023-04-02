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

# NPP

This lab utilized the cudaNPP libraries. These libraries include a lot of functionality for image and video processing.

For this assignment I created a small program that applies a gaussian blur using these libraries. This program takes in an 8-bit pgm image and outputs 
the image with a gaussian blur applied.

The arguments passed to this program are:
```
Call ./npp_assignment {input_file} {output_file}
```

I additionally created a *run.sh* script that runs this npp program two times with two different images.
The output of this file is available in [documentation/npp_run.log](documentation/npp_run.log).

This output shows the time that the gpu spent applying the filter to the image. This can be seen below:
```
Frederich Stine - EN 605.617 - JHU EP
----------------------------------------------------------------
Example runner to show execution of Module 9 npp assignment
----------------------------------------------------------------
This runner runs my example npp code that applies a gaussian
blur to a grayscale image.
This runner runs through two separate grayscale images differing in
size by 512x512 to 5184x3456
Press enter to continue:
NPP gaussian blur: file of 512x512
==16291== NVPROF is profiling process 16291, command: ./npp_assignment images/Lena.pgm images/Lena-gauss.pgm
Perf: Loading image
Perf: Running gaussian filter
Perf: Saving image
==16291== Profiling application: ./npp_assignment images/Lena.pgm images/Lena-gauss.pgm
==16291== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   37.26%  21.793us         1  21.793us  21.793us  21.793us  [CUDA memcpy HtoD]
                   34.90%  20.415us         1  20.415us  20.415us  20.415us  [CUDA memcpy DtoH]
                   27.84%  16.288us         1  16.288us  16.288us  16.288us  void TwoPassFilter32f<unsigned char, int=1, FilterGauss5x532fTwoPassReplicateBorderFunctor<unsigned char, int=1>, int=5>(Image<unsigned char, int=1>, NppiSize, unsigned char)
      API calls:   49.10%  89.682ms         2  44.841ms  4.1980us  89.678ms  cudaMallocPitch
                   48.85%  89.222ms        64  1.3941ms  19.076us  23.267ms  cuLibraryLoadData
                    1.41%  2.5736ms         1  2.5736ms  2.5736ms  2.5736ms  cudaLaunchKernel
                    0.22%  398.35us       447     891ns     110ns  49.133us  cuDeviceGetAttribute
                    0.15%  273.15us         2  136.57us  52.409us  220.74us  cudaMemcpy2D
                    0.11%  199.27us      1149     173ns     110ns  3.6670us  cuGetProcAddress
                    0.08%  142.43us         2  71.213us  9.2770us  133.15us  cudaFree
                    0.04%  75.352us         1  75.352us  75.352us  75.352us  cudaGetDeviceProperties
                    0.03%  55.316us         4  13.829us  8.2360us  18.926us  cuDeviceGetName
                    0.01%  10.350us         1  10.350us  10.350us  10.350us  cudaStreamGetFlags
                    0.00%  7.0230us         1  7.0230us  7.0230us  7.0230us  cuDeviceGetPCIBusId
                    0.00%  4.5690us         1  4.5690us  4.5690us  4.5690us  cudaGetDevice
                    0.00%  2.1330us         3     711ns     410ns  1.2120us  cuInit
                    0.00%  1.6830us         6     280ns     140ns     631ns  cuDeviceGetCount
                    0.00%     960ns         4     240ns     160ns     340ns  cuDeviceTotalMem
                    0.00%     792ns         5     158ns     120ns     280ns  cuDeviceGet
                    0.00%     661ns         2     330ns     290ns     371ns  cudaDeviceGetAttribute
                    0.00%     591ns         4     147ns     111ns     220ns  cuModuleGetLoadingMode
                    0.00%     561ns         4     140ns     120ns     171ns  cuDeviceGetUuid
                    0.00%     441ns         1     441ns     441ns     441ns  cudaGetLastError
                    0.00%     390ns         3     130ns     130ns     130ns  cuDriverGetVersion
Press enter to continue:
NPP gaussian blur: file of 5184x3456
==16305== NVPROF is profiling process 16305, command: ./npp_assignment images/sample.pgm images/sample-gauss.pgm
Perf: Loading image
Perf: Running gaussian filter
Perf: Saving image
==16305== Profiling application: ./npp_assignment images/sample.pgm images/sample-gauss.pgm
==16305== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.53%  5.1314ms         1  5.1314ms  5.1314ms  5.1314ms  [CUDA memcpy DtoH]
                   27.01%  2.2168ms         1  2.2168ms  2.2168ms  2.2168ms  [CUDA memcpy HtoD]
                   10.46%  858.02us         1  858.02us  858.02us  858.02us  void TwoPassFilter32f<unsigned char, int=1, FilterGauss5x532fTwoPassReplicateBorderFunctor<unsigned char, int=1>, int=5>(Image<unsigned char, int=1>, NppiSize, unsigned char)
      API calls:   48.45%  82.871ms        64  1.2949ms  18.916us  20.738ms  cuLibraryLoadData
                   44.12%  75.457ms         2  37.729ms  152.45us  75.305ms  cudaMallocPitch
                    5.03%  8.6062ms         2  4.3031ms  2.2829ms  6.3233ms  cudaMemcpy2D
                    1.50%  2.5645ms         1  2.5645ms  2.5645ms  2.5645ms  cudaLaunchKernel
                    0.44%  760.47us         2  380.24us  184.80us  575.67us  cudaFree
                    0.23%  387.69us       447     867ns     110ns  46.167us  cuDeviceGetAttribute
                    0.13%  228.80us      1149     199ns     120ns  3.0760us  cuGetProcAddress
                    0.04%  73.098us         1  73.098us  73.098us  73.098us  cudaGetDeviceProperties
                    0.03%  57.899us         4  14.474us  8.5160us  18.034us  cuDeviceGetName
                    0.01%  15.279us         1  15.279us  15.279us  15.279us  cudaStreamGetFlags
                    0.00%  5.7410us         1  5.7410us  5.7410us  5.7410us  cuDeviceGetPCIBusId
                    0.00%  5.4800us         1  5.4800us  5.4800us  5.4800us  cudaGetDevice
                    0.00%  3.6670us         3  1.2220us     531ns  1.5730us  cuInit
                    0.00%  1.7630us         6     293ns     151ns     611ns  cuDeviceGetCount
                    0.00%  1.0540us         4     263ns     171ns     311ns  cuDeviceTotalMem
                    0.00%     840ns         5     168ns     120ns     320ns  cuDeviceGet
                    0.00%     680ns         2     340ns     280ns     400ns  cudaDeviceGetAttribute
                    0.00%     611ns         4     152ns     120ns     231ns  cuModuleGetLoadingMode
                    0.00%     592ns         4     148ns     121ns     201ns  cuDeviceGetUuid
                    0.00%     451ns         1     451ns     451ns     451ns  cudaGetLastError
                    0.00%     391ns         3     130ns     110ns     150ns  cuDriverGetVersion
```

The first run processes an image that is 512x512 or .26 MP. This operation happens very quickly on the GPU and the the processing time is comparable 
to the time spent executing the kernel. The second run processes an image that is 5184x3456 or 17.9 MP. This takes 52.9 times longer than the .26 MP image while being 68.34 time larger. This shows that the amount of time processing on the GPU scales very evenly with the size of images to NPP. The memcpy operations for the larger image are much longer than those for the smaller image. This is something to take into consideration if using NPP for processing large images.
I think that to get around this issue, it would probably be best to use NPP processing with streams such that the data transfers could continuously be utilized. 
This operation that I ran is very simple, but performing more computationally heavy chains of operations would make more sense when using NPP. 
This would help to even out the amount of time spend copying data with the amount of time processig data.

Overall NPP is really great library that is an alterative to OpenCV which I have used in the past to perform the exact same operations. I do not have 
that code currently, but would imagine that the times would be comparable (if not longer for npp) in this case due to that large amount of time 
spent copying the image from host to GPU and back.

One of the images that I applied the filter to is available below before and after filtering:

It doesn't seem that these images are viewable on Github markdown - they are pgm images.
If you have the chance you can pull them down and view them in the /images directory of this repo.

## CuGraph

I was unable to utilize the CuGraph library due to the version of Cuda that I am on. I am utilizing Cuda 12.0 and this library was depricated in Cuda 10.

