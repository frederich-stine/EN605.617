all: hello-world blocks grids assignment

assignment: assignment.cu
	nvcc assignment.cu -o assignment 

hello-world: hello-world.cu
	nvcc hello-world.cu -o hello-world

blocks: blocks.cu
	nvcc blocks.cu -o blocks

grids: grids.cu
	nvcc grids.cu -o grids

clean:
	rm -f hello-world blocks grids assignment
