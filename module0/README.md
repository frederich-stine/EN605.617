# Module 0 - Frederich Stine

This module dealt with installing Cuda/OpenCL on the system that we intend to use for this class.

I am using my desktop running Ubuntu 22.04 with a GTX1080.

## Cuda install

- Cuda Toolkit: I installed Cuda toolkit from: `https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04`
- Cuda: I then followed the Ubuntu network directions to install Cuda: `https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04`

## OpenCL install

- The OpenCL libraries were installed with my Cuda installation
- To make them accessible I created two symbolic links:
  - /usr/local/cuda-12.0/targets/x86_64-linux/include/CL to /usr/include
  - /usr/local/cuda-12.0/targets/x86_64-linux/lib/libOpenCL.so to /usr/lib

## Makefile

Last I created a very simple makefile to demonstrate compiling the different examples.
