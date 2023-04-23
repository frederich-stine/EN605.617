all: simple assignment

simple: simple.cpp simple.cl
	g++ simple.cpp -o simple -l OpenCL

assignment: assignment.cpp simple.cl
	g++ assignment.cpp -o assignment -l OpenCL

clean:
	rm -f assignment simple