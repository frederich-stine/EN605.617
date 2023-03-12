all: stream_example assignment

stream_example: stream_example.cu
	nvcc $^ -o $@

assignment: assignment.cu
	nvcc $^ -o $@

clean:
	rm -f assignment stream_example
