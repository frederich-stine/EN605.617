all: cublas_example cufft_example curand_example \
	cusolver_example assignment_1 assignment_2

cublas_example: cublas_example.cu
	nvcc $^ -o $@ -lcublas

cufft_example: cufft_example.cu
	nvcc $^ -o $@ -lcufft

curand_example: curand_example.cu
	nvcc $^ -o $@ -lcurand

cusolver_example: cusolver_example.cu
	nvcc $^ -o $@ -lcusolver -lcublas

assignment_1: assignment_1.cu
	nvcc $^ -o $@ -lcurand

assignment_2: assignment_2.cu
	nvcc $^ -o $@ -lcufft

# My version of CUDA is too new for this example
cusparse_example: cusparse_example.cu
	nvcc $^ -o $@ -lcublas -lcusparse

clean: 
	rm -rf cublas_example cufft_example curand_example \
		cusolver_example assignment_1 assignment_2
