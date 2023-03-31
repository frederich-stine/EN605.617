#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <stdio.h>
#include <stdint.h>

void thrust_arith (int op, uint32_t vect_size);

int main (int argc, char** argv) {
	
	// Prints out a help menu if not enough params are passed
	if (argc != 3) {
		printf("Call ./thrust_assignment {vector_size} {operation}}\n");
		printf("Operations: \n");
		printf("    0: Add\n");
		printf("    1: Sub\n");
		printf("    2: Mult\n");
		printf("    3: Mod\n");
		exit(0);
	}

	int vect_size = atoi(argv[1]);
	uint32_t op = atoi(argv[2]);

	if (op > 3) {
		printf("Error: Incorrect operation provided\n");
		exit(0);
	}

	thrust_arith(op, vect_size);

}

void thrust_arith (int op, uint32_t vect_size) {
	thrust::host_vector<int> h1(vect_size);
	thrust::host_vector<int> h2(vect_size);

	for (int i=0; i<vect_size; i++) {
		h1[i] = (int) i;
		h2[i] = (int) vect_size-i;
	}

	thrust::device_vector<int> d1 = h1;
	thrust::device_vector<int> d2 = h2;
	thrust::device_vector<int> d3(vect_size);
	thrust::fill(d3.begin(), d3.end(), 0);

	switch (op) {
	case 0:
		thrust::transform(d1.begin(), d1.end(), d2.begin(),\
				d3.begin(), thrust::plus<int>());
		break;
	case 1:
		thrust::transform(d1.begin(), d1.end(), d2.begin(),\
				d3.begin(), thrust::minus<int>());
		break;
	case 2:
		thrust::transform(d1.begin(), d1.end(), d2.begin(),\
				d3.begin(), thrust::multiplies<int>());
		break;
	case 3:
		thrust::transform(d1.begin(), d1.end(), d2.begin(),\
				d3.begin(), thrust::modulus<int>());
		break;
	}

	for (int i=0; i<vect_size; i++) {
		printf("Array 1[%d]: %4d :", i, (int)d1[i]);
		printf("Array 2[%d]: %4d :", i, (int)d2[i]);
		printf("Result[%d]: %4d\n", i, (int)d3[i]);
	}
}

