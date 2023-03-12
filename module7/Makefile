all: stream_example assignment stretch_problem

stream_example: stream_example.cu
	nvcc $^ -o $@

assignment: assignment.cu
	nvcc $^ -o $@

stretch_problem: stretch_problem.cu
	nvcc $^ -o $@

clean:
	rm -f assignment stream_example stretch_problem
