# Module 8 - CUDA Libraries

## cuRand and cuFFT

For this lab I chose to demonstrate the cuRand and cuFFT libraries. I specifically chose cuFFT because part of my project will be using this library 
to show the different in audio spectrum before and after applying FIR filters to an audio file. Instead of creating one file, I created two separate 
files for this lab *assignment_1.cu* and *assignment_2.cu*. The first is the cuRand example and the second is the cuFFT example.

I chose to use the host APIs provided by these libraries instead of using the device side functionality. This was interesting because for cuRand 
it was the opposite of what the example code did.

For cuRand I created a simple program that generates a variable number of uniform random values.

For the cuFFT program I chose to get some knowledge that would help me with my project. For my project I want to apply FIR filters and FFTs to audio files
using the gpu. To get started on doing this I created a program that reads in samples from a WAV file and runs a very simple FFT on them. The audio file 
that I chose is just a pure sin wave at 400Hz. Using the FFT, we can see that this is the dominant spectrum in the file. My FFT example can run with a 
variable size of FFTs passed as a user parameter.

The arguments passed to the cuRand example are: 
```
Simple cuRand example host calls
    Call ./assignment {threadCount}
```

The arguments passed to the cuFFT example are:
```
Simple audio spectrum FFT example
    Call ./assignment {FFT_SIZE} 
```

I additionally created a run.sh script that runs both of these files with different parameters. Nvprof is used for some timing characteristics of the 
program. I am not trying to show an inherent speedup here, but want to see what calls are being made by the libraries. The full log of the *run.sh* scipt 
is available in [documentation/full_run.log](documentation/full_run.log)

Here is the result of running the random example, I have cut out a lot of the output for brevity.
```
Curand example
Generating 500 uniform random numbers
==15774== NVPROF is profiling process 15774, command: ./assignment_1 500
Random 0: 0.070891
Random 1: 0.099631
Random 2: 0.618459
Random 3: 0.488424
Random 4: 0.471553
Random 5: 0.879565
Random 6: 0.504235
Random 7: 0.587636
Random 493: 0.444506
Random 494: 0.799413
Random 495: 0.904474
Random 496: 0.541942
Random 497: 0.118399
Random 498: 0.604295
Random 499: 0.639104
==15774== Profiling application: ./assignment_1 500
==15774== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.33%  484.13us         1  484.13us  484.13us  484.13us  void generate_seed_pseudo<rng_config<curandStateXORWOW, curandOrdering=101>>(__int64, __int64, __int64, curandOrdering, curandStateXORWOW*, unsigned int*)
                    0.41%  1.9840us         1  1.9840us  1.9840us  1.9840us  void gen_sequenced<curandStateXORWOW, float, int, __operator_&__(float curand_uniform_noargs<curandStateXORWOW>(curandStateXORWOW*, int)), rng_config<curandStateXORWOW, curandOrdering=101>>(curandStateXORWOW*, float*, unsigned long, unsigned long, int)
                    0.26%  1.2800us         1  1.2800us  1.2800us  1.2800us  [CUDA memcpy DtoH]
      API calls:   88.57%  108.57ms         1  108.57ms  108.57ms  108.57ms  cudaMallocHost
                   10.03%  12.292ms        11  1.1175ms  17.313us  4.6880ms  cuLibraryLoadData
                    0.40%  485.84us         1  485.84us  485.84us  485.84us  cudaDeviceSynchronize
                    0.31%  384.25us         1  384.25us  384.25us  384.25us  cudaFreeHost
                    0.26%  318.30us         6  53.049us     291ns  269.66us  cudaFree
                    0.23%  277.55us       225  1.2330us     110ns  69.521us  cuDeviceGetAttribute
                    0.06%  77.465us       383     202ns     120ns     962ns  cuGetProcAddress
                    0.05%  62.237us         2  31.118us  6.9830us  55.254us  cudaMalloc
                    0.03%  42.460us         2  21.230us  14.858us  27.602us  cuDeviceGetName
                    0.02%  30.177us         2  15.088us  4.3180us  25.859us  cudaLaunchKernel
                    0.01%  15.739us         1  15.739us  15.739us  15.739us  cudaMemcpy
                    0.01%  8.1750us         1  8.1750us  8.1750us  8.1750us  cuDeviceGetPCIBusId
                    0.01%  6.5020us         1  6.5020us  6.5020us  6.5020us  cudaGetDevice
                    0.00%  2.7860us         3     928ns     421ns  1.7540us  cudaDeviceGetAttribute
                    0.00%  2.7660us         4     691ns     271ns  1.5830us  cuDeviceGetCount
                    0.00%  1.8330us         1  1.8330us  1.8330us  1.8330us  cuInit
                    0.00%     992ns         3     330ns     141ns     611ns  cuDeviceGet
                    0.00%     941ns         2     470ns     440ns     501ns  cuDeviceTotalMem
                    0.00%     701ns         4     175ns     130ns     271ns  cudaGetLastError
                    0.00%     691ns         2     345ns     130ns     561ns  cuModuleGetLoadingMode
                    0.00%     482ns         2     241ns     151ns     331ns  cuDeviceGetUuid
                    0.00%     151ns         1     151ns     151ns     151ns  cuDriverGetVersion
```

We can see that random numbers are generated between 0-1 in a uniform distribution. We also see two device functions called which correlates to the 
two host calls that were made to seed and generate the random variables.

Here is the result of running the fft example, I have cut out a lot of the output for brevity.
```
Audio FFT example
Running FFT with FFT size of 64
==15802== NVPROF is profiling process 15802, command: ./assignment_2 64
Sample rate: 44100
Bits per sample: 16
Num channels: 1
FFT Result: Frequency: 0.000000: C: 603120.000000: I: 0.000000 
    Magnitude: 603120.000000: Magnitude dB: 115.608070
FFT Result: Frequency: 689.062500: C: -400917.250000: I: -297988.031250 
    Magnitude: 499531.281250: Magnitude dB: 113.971252
FFT Result: Frequency: 1378.125000: C: -59025.621094: I: -98025.531250 
    Magnitude: 114424.773438: Magnitude dB: 101.170395
FFT Result: Frequency: 2067.187500: C: -19741.453125: I: -61219.082031 
    Magnitude: 64323.410156: Magnitude dB: 96.167381
FFT Result: Frequency: 2756.250000: C: -7079.578125: I: -44728.406250 
    Magnitude: 45285.214844: Magnitude dB: 93.119125
FFT Result: Frequency: 3445.312500: C: -1385.718750: I: -35184.500000 
    Magnitude: 35211.777344: Magnitude dB: 90.933762
FFT Result: Frequency: 4134.375000: C: 1660.523926: I: -28907.673828 
    Magnitude: 28955.326172: Magnitude dB: 89.234573
FFT Result: Frequency: 4823.437500: C: 3480.192139: I: -24433.195312 
    Magnitude: 24679.804688: Magnitude dB: 87.846832
FFT Result: Frequency: 5512.500000: C: 4658.419922: I: -21063.992188 
    Magnitude: 21572.960938: Magnitude dB: 86.678200
FFT Result: Frequency: 6201.562500: C: 5455.962891: I: -18431.843750 
    Magnitude: 19222.392578: Magnitude dB: 85.676147
FFT Result: Frequency: 6890.625000: C: 6029.536621: I: -16283.212891 
    Magnitude: 17363.707031: Magnitude dB: 84.792854
FFT Result: Frequency: 7579.687500: C: 6451.500000: I: -14516.611328 
    Magnitude: 15885.649414: Magnitude dB: 84.020096
FFT Result: Frequency: 8268.750000: C: 6767.907227: I: -13012.725586 
    Magnitude: 14667.500977: Magnitude dB: 83.327118
FFT Result: Frequency: 8957.812500: C: 7021.134277: I: -11716.607422 
    Magnitude: 13659.253906: Magnitude dB: 82.708542
FFT Result: Frequency: 9646.875000: C: 7204.860840: I: -10590.597656 
    Magnitude: 12809.011719: Magnitude dB: 82.150314
FFT Result: Frequency: 10335.937500: C: 7371.073242: I: -9582.720703 
    Magnitude: 12089.716797: Magnitude dB: 81.648323
FFT Result: Frequency: 11025.000000: C: 7499.000488: I: -8679.000000 
    Magnitude: 11469.962891: Magnitude dB: 81.191246
FFT Result: Frequency: 11714.062500: C: 7601.282227: I: -7868.755371 
    Magnitude: 10940.603516: Magnitude dB: 80.780830
FFT Result: Frequency: 12403.125000: C: 7687.803223: I: -7127.171875 
    Magnitude: 10483.267578: Magnitude dB: 80.409935
FFT Result: Frequency: 13092.187500: C: 7760.520996: I: -6440.974121 
    Magnitude: 10085.228516: Magnitude dB: 80.073715
FFT Result: Frequency: 13781.250000: C: 7822.704102: I: -5802.190430 
    Magnitude: 9739.615234: Magnitude dB: 79.770836
FFT Result: Frequency: 14470.312500: C: 7873.773438: I: -5202.247559 
    Magnitude: 9437.144531: Magnitude dB: 79.496811
FFT Result: Frequency: 15159.375000: C: 7915.761230: I: -4636.919434 
    Magnitude: 9173.892578: Magnitude dB: 79.251076
FFT Result: Frequency: 15848.437500: C: 7954.371094: I: -4101.919434 
    Magnitude: 8949.735352: Magnitude dB: 79.036201
FFT Result: Frequency: 16537.500000: C: 7977.580078: I: -3589.992188 
    Magnitude: 8748.132812: Magnitude dB: 78.838303
FFT Result: Frequency: 17226.562500: C: 8007.418945: I: -3102.838867 
    Magnitude: 8587.570312: Magnitude dB: 78.677406
FFT Result: Frequency: 17915.625000: C: 8027.735840: I: -2631.345703 
    Magnitude: 8447.989258: Magnitude dB: 78.535065
FFT Result: Frequency: 18604.687500: C: 8040.925781: I: -2170.275391 
    Magnitude: 8328.660156: Magnitude dB: 78.411499
FFT Result: Frequency: 19293.750000: C: 8056.966797: I: -1725.869141 
    Magnitude: 8239.741211: Magnitude dB: 78.318268
FFT Result: Frequency: 19982.812500: C: 8065.303711: I: -1285.281250 
    Magnitude: 8167.072266: Magnitude dB: 78.241333
FFT Result: Frequency: 20671.875000: C: 8075.402344: I: -852.066406 
    Magnitude: 8120.230469: Magnitude dB: 78.191368
FFT Result: Frequency: 21360.937500: C: 8080.937500: I: -427.343750 
    Magnitude: 8092.229004: Magnitude dB: 78.161362
FFT Result: Frequency: 22050.000000: C: 8082.000000: I: 0.000000 
    Magnitude: 8082.000000: Magnitude dB: 78.150375
==15802== Profiling application: ./assignment_2 64
==15802== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   96.81%  258.08us         4  64.520us     224ns  257.19us  [CUDA memcpy HtoD]
                    1.66%  4.4160us         1  4.4160us  4.4160us  4.4160us  void vector_fft<unsigned int=32, unsigned int=8, unsigned int=32, unsigned int=10, padding_t=1, twiddle_t=0, loadstore_modifier_t=2, layout_t=0, unsigned int, float>(kernel_arguments_t<unsigned int>)
                    1.03%  2.7520us         1  2.7520us  2.7520us  2.7520us  void postprocess_kernel<float, unsigned int, loadstore_modifier_t=2>(real_complex_args_t<unsigned int>)
                    0.50%  1.3430us         1  1.3430us  1.3430us  1.3430us  [CUDA memcpy DtoH]
      API calls:   88.97%  92.090ms         2  46.045ms  4.0870us  92.086ms  cudaMallocHost
                    8.43%  8.7239ms        14  623.14us  15.439us  6.8677ms  cuLibraryLoadData
                    0.97%  1.0078ms         4  251.95us  2.9460us  913.63us  cudaFree
                    0.39%  398.72us         3  132.91us  3.4170us  338.74us  cuMemcpyHtoD
                    0.36%  370.58us         2  185.29us  6.3830us  364.20us  cudaFreeHost
                    0.27%  279.18us       231  1.2080us     110ns  67.637us  cuDeviceGetAttribute
                    0.20%  210.38us         2  105.19us  82.054us  128.32us  cuModuleLoadData
                    0.07%  73.851us       383     192ns     120ns  1.0320us  cuGetProcAddress
                    0.06%  65.604us         1  65.604us  65.604us  65.604us  cuMemAlloc
                    0.06%  61.216us         3  20.405us  14.658us  28.434us  cuDeviceGetName
                    0.06%  60.224us         3  20.074us  2.3850us  52.198us  cudaMalloc
                    0.04%  40.837us         1  40.837us  40.837us  40.837us  cuMemFree
                    0.03%  29.805us         2  14.902us  12.864us  16.941us  cudaMemcpy
                    0.02%  22.743us         2  11.371us  8.7660us  13.977us  cuModuleUnload
                    0.02%  20.649us         2  10.324us  4.2080us  16.441us  cuLaunchKernel
                    0.01%  14.668us         1  14.668us  14.668us  14.668us  cuMemGetInfo
                    0.01%  8.4060us         1  8.4060us  8.4060us  8.4060us  cuDeviceGetPCIBusId
                    0.00%  4.5890us         1  4.5890us  4.5890us  4.5890us  cudaStreamIsCapturing
                    0.00%  3.4950us        20     174ns     120ns     751ns  cuFuncGetAttribute
                    0.00%  3.4470us         2  1.7230us     561ns  2.8860us  cuModuleGetFunction
                    0.00%  2.8450us         3     948ns     481ns  1.7930us  cudaGetDevice
                    0.00%  2.8250us         4     706ns     290ns  1.5730us  cuDeviceGetCount
                    0.00%  1.4720us         2     736ns     260ns  1.2120us  cuOccupancyMaxActiveBlocksPerMultiprocessor
                    0.00%  1.3730us         1  1.3730us  1.3730us  1.3730us  cuInit
                    0.00%  1.3020us         5     260ns     150ns     441ns  cuCtxPushCurrent
                    0.00%  1.2530us         2     626ns     371ns     882ns  cuPointerGetAttribute
                    0.00%  1.1620us         1  1.1620us  1.1620us  1.1620us  cudaSetDevice
                    0.00%  1.1210us         3     373ns     130ns     721ns  cuDeviceGet
                    0.00%  1.0320us         6     172ns     121ns     320ns  cuCtxGetCurrent
                    0.00%     901ns         5     180ns     150ns     220ns  cuCtxPopCurrent
                    0.00%     762ns         2     381ns     311ns     451ns  cuModuleGetGlobal
                    0.00%     732ns         2     366ns     271ns     461ns  cuDeviceTotalMem
                    0.00%     611ns         2     305ns     130ns     481ns  cuModuleGetLoadingMode
                    0.00%     521ns         2     260ns     150ns     371ns  cuDeviceGetUuid
                    0.00%     491ns         2     245ns     150ns     341ns  cudaGetFuncBySymbol
                    0.00%     150ns         1     150ns     150ns     150ns  cuDriverGetVersion
                    0.00%     120ns         1     120ns     120ns     120ns  cuCtxGetDevice
```

Here we can see that two device calls were made for running this FFT on the kernel. This was nicely abstracted away from the user (me) through the API. 
We can also see that the magnitude at 0 is higher than at all other points in the result. Since 440hz does not evenly divide into the 44.1khz sample rate 
we see a smearing effect where the magnitude shows up across all of the FFT segments. This gets cleared as the size of the FFT window increases. 
Since the window here was relatively small, the first two buckets are higher than the rest. One other thing to note is that the magnitude here is high. 
I did not normalize the data from the audio file before processing and this would result with easier to process results.

When applying this to my project I will try to move all of the calculations that I made to the GPU. This includes normalization, conversion from int 
to float, magnitude calculation, db calculation. I will additionally use a much larger batch size such that I can process a whole file at one time.

There will be some overlap between this and my project in that the same type of kernel and wav file audio processing for spectrography will be utilized.

After completing this assignment I understand how to use these useful libraries included by CUDA. This is a real timesaver when developing larger projects 
and will be something that I utilize in the future for my project and professional projects.
