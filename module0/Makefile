all: cuda opencl cpp

opencl:
	g++ hello_world_cl.c -o hello_world_cl -l OpenCL

cuda:
	nvcc hello-world.cu -o hello_world_cuda

cpp:
	g++ helloworld.cpp -o hello_world
