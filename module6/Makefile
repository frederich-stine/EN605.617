all: register assignment

register: register.cu
	nvcc $^ -o $@

assignment: assignment.cu
	nvcc $^ -o $@

clean: 
	rm -f register assignment
