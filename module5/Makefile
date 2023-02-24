all: constant_memory constant_memory2 \
	shared_memory shared_memory2 \
	assignment

constant_memory: constant_memory.cu
	nvcc $^ -o $@

constant_memory2: constant_memory2.cu
	nvcc $^ -o $@

shared_memory: shared_memory.cu
	nvcc $^ -o $@

shared_memory2: shared_memory2.cu
	nvcc $^ -o $@

assignment: assignment.cu
	nvcc $^ -o $@

clean:
	rm -f assignment const_test constant_memory \
		constant_memory2 shared_memory \
		shared_memory2
