all: Convolution vector_example OpenCLInfo assignment

Convolution: Convolution.cpp
	g++ Convolution.cpp -o Convolution -l OpenCL

vector_example: vector_example.cpp
	g++ vector_example.cpp -o vector_example -l OpenCL

OpenCLInfo: OpenCLInfo.cpp
	g++ OpenCLInfo.cpp -o OpenCLInfo -l OpenCL

assignment: assignment.cpp
	g++ assignment.cpp -o assignment -l OpenCL

clean:
	rm -f Convolution vector_example OpenCLInfo assignment
