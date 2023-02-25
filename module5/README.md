# Module 5 - Frederich Stine

## Shared and Constant Memory

To demonstrate the ability to use both shared and constant memory I created four separate kernels.
- One that copies data into shared memory before performing calculations
- One that uses shared memory as local memory inside of the kernel
- One that copies data to a constant array with CudaMemcpyToSymbol
- One that uses some hardcoded constant values

All four of these kernels perform the same operations. The first three perform the same operations on the same quantity of data.
The final one that uses hardcoded values uses 4 pre-set values instead of utilizing an array of 1024 initialized values.

All of the kernels run with a threadCount of 1024. This limit is imposed by the constant functions where
memory has to be statically declared. The shared memory kernels can be run with variable threadCounts by 
allowing this as a user input, but was not done for this assignment as would not provide a valid comparison.

As with previous assignments, this one *assignment.cu* file contains all of the functionality and uses 
command line arguments to choose between different functions. This also allows for the use of a variable 
block size. The help menu for this is below:

```
Call ./assignment {blockSize} {operation}
Operations: 
    0: Copy to shared Memory
    1: Shared memory for local
    2: Copy to constant memory
    3: Constant memory only
```

Using this knowledge we can compare the performance again using *nvprof*. It will be good to note that some 
of the calls require differing amounts of memcpy and allocation. I still enjoy the fine amount of detail that 
nvprof allows over simple timing of single or grouped functions.

I additionally created a *run.sh* script which runs all four kernels with two different block sizes. The output 
of half of this with nvprof is shown below. For sake of brevity I have removed the textural output from the program 
and only left the timing statistics.






