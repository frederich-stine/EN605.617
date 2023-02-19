# Module 4 - Frederich Stine

## Paged vs Pinned Memory 

To demonstrate the different in performance between paged and pinned memory in Cuda I created two identical functions
that operated with paged memory and pinned memory. These functions both allocated and initialized data and then ran a simple CUDA kernel that
both added, subtracted, multiplied, and applied the modulo operator. To allocate paged memory I used the standard C `malloc()` function
and to allocated pinned memory I used the `CudaMallocHost()` function. This code is written in **assignment.cu** and either function
can be invoked from the command line using command line arguments. This program takes 3 command line arguments:

```
Call ./assignment {numThreads} {blockSize} {operation}
Operations: 
    0: Paged Memory
    1: Pinned Memory
```

I changed how I timed the execution of functions for this module 4 assignment. In module 3 I used simple C timers to determine the amount of time 
that functions took. Instead of using that technique, this time I used the nvprof (Nvidia Profiler) utility included with the CUDA toolkit. 
This profiler was an excellent solution as it was both more accurate that my previous solution and more detailed. This shows the breadown im 
time taken by all CUDA functions called within the program. Since I wanted to compare the time `CudaMemcpy` takes, I was able to view 
the overall time used by this function. I only called this within the testing code so the value that we see is the value related to paged/pinned memory.  

An example output is shown below:
```
Press enter to continue:Paged memory - 16777216 threads - block size 128
==63695== NVPROF is profiling process 63695, command: ./assignment 16777216 128 0
==63695== Profiling application: ./assignment 16777216 128 0
==63695== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   55.77%  19.317ms         1  19.317ms  19.317ms  19.317ms  [CUDA memcpy DtoH]
                   41.36%  14.326ms         2  7.1629ms  6.9649ms  7.3608ms  [CUDA memcpy HtoD]
                    2.87%  993.16us         1  993.16us  993.16us  993.16us  gpu_all_arith(int*, int*, int*)
      API calls:   63.51%  68.532ms         3  22.844ms  84.729us  68.350ms  cudaMalloc
                   32.75%  35.340ms         3  11.780ms  7.0551ms  20.870ms  cudaMemcpy
                    3.50%  3.7717ms         3  1.2572ms  165.34us  1.8117ms  cudaFree
                    0.10%  104.90us       114     920ns     110ns  45.996us  cuDeviceGetAttribute
                    0.08%  86.964us         1  86.964us  86.964us  86.964us  cuLibraryLoadData
                    0.04%  42.410us         1  42.410us  42.410us  42.410us  cudaLaunchKernel
                    0.02%  16.211us         1  16.211us  16.211us  16.211us  cuDeviceGetName
                    0.00%  3.6570us         1  3.6570us  3.6570us  3.6570us  cuDeviceGetPCIBusId
                    0.00%  1.2530us         3     417ns     170ns     812ns  cuDeviceGetCount
                    0.00%     521ns         1     521ns     521ns     521ns  cuModuleGetLoadingMode
                    0.00%     411ns         2     205ns     130ns     281ns  cuDeviceGet
                    0.00%     290ns         1     290ns     290ns     290ns  cuDeviceTotalMem
                    0.00%     160ns         1     160ns     160ns     160ns  cuDeviceGetUuid
Pinned memory - 16777216 threads - block size 128
==63708== NVPROF is profiling process 63708, command: ./assignment 16777216 128 1
==63708== Profiling application: ./assignment 16777216 128 1
==63708== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   63.25%  10.263ms         2  5.1314ms  5.1277ms  5.1351ms  [CUDA memcpy HtoD]
                   30.55%  4.9571ms         1  4.9571ms  4.9571ms  4.9571ms  [CUDA memcpy DtoH]
                    6.20%  1.0056ms         1  1.0056ms  1.0056ms  1.0056ms  gpu_all_arith(int*, int*, int*)
      API calls:   78.55%  147.41ms         3  49.136ms  17.185ms  112.91ms  cudaMallocHost
                   10.34%  19.401ms         3  6.4670ms  5.5898ms  7.8955ms  cudaFreeHost
                    8.69%  16.315ms         3  5.4383ms  5.1444ms  6.0077ms  cudaMemcpy
                    1.99%  3.7332ms         3  1.2444ms  116.52us  1.8138ms  cudaFree
                    0.23%  424.25us         3  141.42us  82.445us  251.22us  cudaMalloc
                    0.09%  163.38us       114  1.4330us     230ns  67.838us  cuDeviceGetAttribute
                    0.08%  143.65us         1  143.65us  143.65us  143.65us  cuLibraryLoadData
                    0.01%  26.750us         1  26.750us  26.750us  26.750us  cudaLaunchKernel
                    0.01%  25.097us         1  25.097us  25.097us  25.097us  cuDeviceGetName
                    0.00%  7.8650us         1  7.8650us  7.8650us  7.8650us  cuDeviceGetPCIBusId
                    0.00%  1.9930us         3     664ns     330ns  1.1220us  cuDeviceGetCount
                    0.00%     972ns         2     486ns     281ns     691ns  cuDeviceGet
                    0.00%     561ns         1     561ns     561ns     561ns  cuModuleGetLoadingMode
                    0.00%     471ns         1     471ns     471ns     471ns  cuDeviceTotalMem
                    0.00%     321ns         1     321ns     321ns     321ns  cuDeviceGetUuid
```

The results from nvprof here directly show the relationship that I was trying to show. The top run is using paged memory
and for the same operation, the copies took anywhere from 50% to almost 200% longer. It is interesting to note here that CudaMallocHost 
shows a very long run time of 49.136ms. I did some research into this and found that this is not accurate. Since CudaMallocHost is the first 
CUDA function running in this example, the time included in the 49ms includes initializing the CUDA environment. That being said, the minimum time 
for this function is still 17ms, which is much longer than a typical allocation operation would take (look at cudaMalloc). This is some 
of the tradeoff that comes with paged vs pinned memory here. Pinned memory allocation can be slower in the Linux kernel, but once it is set up 
in the CUDA environment data transfers are much faster.

The full output of the run.sh script is available in [documentation/full_run.log](documentation/full_run.log) and shows multiple data sizes and block sizes.

## Caesar Cipher

I was able to write a kernel to carry out a Caesar Cipher for both capital and lowercase letters. I put this in a separate file from the first 
tests, as there was not a large overlap of shared functionality.  In order to fully test this Kernel I wrote some simple code for File I/O to read 
in a text file and write out the results from running the kernel.I wanted to do this so that I could get a large enough datasize to show this 
properly working on the GPU. I used a e-book text file version of Romeo and Juliet to test this cipher from Project Gutenberg. This file is 
available in the repository to test this CUDA kernel.

As with the first program, this program takes in command line arguments. These are highlighted below:

```
Call ./caesar_cipher {blockSize} {input_file} {output_file} {rot}
```

For testing I ran the code through a Caesar Cipher with a rotation of 13 ROT13. The results of this are available in [documentation/romeo_and_juliet_encrypted.txt](documentation/romeo_and_juliet_encrypted.txt).

A small snippet is shown below:

### Original:
```
The Project Gutenberg eBook of Romeo and Juliet, by William Shakespeare

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook.

Title: Romeo and Juliet

Author: William Shakespeare

Release Date: November, 1998 [eBook #1513]
[Most recently updated: May 11, 2022]

Language: English


Produced by: the PG Shakespeare Team, a team of about twenty Project Gutenberg volunteers.

*** START OF THE PROJECT GUTENBERG EBOOK ROMEO AND JULIET ***




THE TRAGEDY OF ROMEO AND JULIET



by William Shakespeare
```

### Encrypted
```
Gur Cebwrpg Thgraoret rObbx bs Ebzrb naq Whyvrg, ol Jvyyvnz Funxrfcrner

Guvf rObbx vf sbe gur hfr bs nalbar naljurer va gur Havgrq Fgngrf naq
zbfg bgure cnegf bs gur jbeyq ng ab pbfg naq jvgu nyzbfg ab erfgevpgvbaf
jungfbrire. Lbh znl pbcl vg, tvir vg njnl be er-hfr vg haqre gur grezf
bs gur Cebwrpg Thgraoret Yvprafr vapyhqrq jvgu guvf rObbx be bayvar ng
jjj.thgraoret.bet. Vs lbh ner abg ybpngrq va gur Havgrq Fgngrf, lbh
jvyy unir gb purpx gur ynjf bs gur pbhagel jurer lbh ner ybpngrq orsber
hfvat guvf rObbx.

Gvgyr: Ebzrb naq Whyvrg

Nhgube: Jvyyvnz Funxrfcrner

Eryrnfr Qngr: Abirzore, 1998 [rObbx #1513]
[Zbfg erpragyl hcqngrq: Znl 11, 2022]

Ynathntr: Ratyvfu


Cebqhprq ol: gur CT Funxrfcrner Grnz, n grnz bs nobhg gjragl Cebwrpg Thgraoret ibyhagrref.

*** FGNEG BS GUR CEBWRPG THGRAORET ROBBX EBZRB NAQ WHYVRG ***




GUR GENTRQL BS EBZRB NAQ WHYVRG



ol Jvyyvnz Funxrfcrner
```

This can be run on any arbitrary input - I also tested encrypting my own Makefile.

## Stretch Problem

I will briefly discuss the good parts and parts for improvement in this code:

- Good parts:
  - Program takes in command line arguments for NUM_THREADS and BLOCK_SIZE correctly
  - Program properly allocates paged and pinned memory
  - Program properly calls timing functions and uses synchronization for asynchronous dispatch
  - Program frees all heap allocated and gpu allocated memory
- Improvements:
  - There is one main improvement for this program: The timing is done around the kernel execution. Since we are trying to view the change in speed of the `cudaMemcpy` function, we should be timing around this function. There should be no change in kernel execution across the two different memory types.
  - A second improvement or general comment is that we implemented the Caesar cipher slightly differently. I do not think that either way is incorrect - if anything this implementation is more secure.

How to fix the timing issue:
Original code:
```c
	 cudaMalloc((void **)&gpu_text, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_key, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_result, array_size_in_bytes);

	 /* Copy the CPU memory to the GPU memory */ 
	 cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice); 
	 cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

	 /* Designate the number of blocks and threads */ 
	 const unsigned int num_blocks = array_size/threads_per_block; 
	 const unsigned int num_threads = array_size/num_blocks;

	 /* Execute the encryption kernel and keep track of start and end time for duration */ 
	 float duration = 0; 
	 cudaEvent_t start_time = get_time();

	 encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);

	 cudaEvent_t end_time = get_time(); 
	 cudaEventSynchronize(end_time); 
	 cudaEventElapsedTime(&duration, start_time, end_time);

	 /* Copy the changed GPU memory back to the CPU */ 
	 cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);

	 printf("Pageable Transfer- Duration: %fmsn\n", duration); 
	 print_encryption_results(cpu_text, cpu_key, cpu_result, array_size);
```

Modified Code:
```c
	 cudaMalloc((void **)&gpu_text, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_key, array_size_in_bytes); 
	 cudaMalloc((void **)&gpu_result, array_size_in_bytes);

	 float duration = 0; 
	 cudaEvent_t start_time = get_time();

	 /* Copy the CPU memory to the GPU memory */ 
	 cudaMemcpy( gpu_text, cpu_text, array_size_in_bytes, cudaMemcpyHostToDevice); 
	 cudaMemcpy( gpu_key, cpu_key, array_size_in_bytes, cudaMemcpyHostToDevice);

	 cudaEvent_t end_time = get_time(); 
	 cudaEventSynchronize(end_time); 
	 cudaEventElapsedTime(&duration, start_time, end_time);

	 printf("Pageable Transfer- Duration: %fmsn\n", duration); 

	 /* Designate the number of blocks and threads */ 
	 const unsigned int num_blocks = array_size/threads_per_block; 
	 const unsigned int num_threads = array_size/num_blocks;

	 encrypt<<<num_blocks, num_threads>>>(gpu_text, gpu_key, gpu_result);
     
     start_time = get_time();

	 /* Copy the changed GPU memory back to the CPU */ 
	 cudaMemcpy( cpu_result, gpu_result, array_size_in_bytes, cudaMemcpyDeviceToHost);
     
     end_time = get_time();
     cudaEventSynchronize(end_time);
     cudaEventElapsedTime(&duration, start_time, end_time);

	 printf("Pageable Transfer- Duration: %fmsn\n", duration); 
	 print_encryption_results(cpu_text, cpu_key, cpu_result, array_size);
```
