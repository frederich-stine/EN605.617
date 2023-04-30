
// Simple kernel for square - from example
__kernel void square(__global float* buffer)
{
	size_t id = get_global_id(0);
	buffer[id] = buffer[id] * buffer[id];
}

// Kernel that does an average on a 1d array
__kernel void average(__global float* buffer, __global float* bufOut, const float width, const int offset) {
	float value = 0.0;
	for (int i=0; i<width; i++) {
		value += buffer[i];
	}

	bufOut[offset] = value/width;
}
