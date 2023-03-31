all: thrust_assignment module3_assignment

thrust_assignment: thrust_assignment.cu
	nvcc $^ -o $@

module3_assignment: module3_assignment.cu
	nvcc $^ -o $@

clean:
	rm -f thrust_assignment module3_assignment
