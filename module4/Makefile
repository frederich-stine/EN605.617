all: global_memory host_memory assignment caesar_cipher

global_memory: global_memory.cu
	nvcc global_memory.cu -o global_memory

host_memory: host_memory.cu
	nvcc host_memory.cu -o host_memory

assignment: assignment.cu
	nvcc assignment.cu -o assignment

caesar_cipher: caesar_cipher.cu
	nvcc caesar_cipher.cu -o caesar_cipher

clean:
	rm -f global_memory host_memory assignment caesar_cipher
