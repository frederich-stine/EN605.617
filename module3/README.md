# Module 3 Assignment - Frederich Stine

## Arithmetic Cuda implementations

To implement the four required arithmetic operations I created four different kernels in the same assignment.cu file.
These operators are called based on the third command line argument passed to the file.

0 - Add  
1 - Sub  
2 - Mult  
3 - Mod  

More information on how to run this properly can be seen in the provided **run.sh** script.

The first two command line arguments are *number of threads* and *block size*. Full output on how to run this can be seen by 
executing the assignment executable without any arguments.

```
Call ./assignment {numThreads} {blockSize} {operation}
Operations:
    0: Add
    1: Sub
    2: Mult
    3: Mod
    4: Branch Compare
```

A full output of the run.sh script is included in [documentation/run.log](documentation/run.log), and as such I will not include the output in this README.

## Conditional Branch Cuda implementation

To test the effects of conditional branching I created some code to time a kernel that included a lot of conditional branching.
I compared this kernel to a kernel that did not include conditional branching as well as the same conditional algorithm running in series on the CPU.
When timing I ran into a few issues. Since the kernel execution is asynchronous, the timing needs to end after the memory is copied back off the GPU to the CPU.
I did not see as big of a contrast between the conditional branching kernel and both the normal CPU code and non-branching kernel as I expected.
I have though of a couple of possiblities as to why this is.

1. The compiler was able to optimize the branches out of my kernel - unable to confirm this
2. The dataset was not large enough to indicate a large enough change in execution time.
3. The GPU driver was able to optimize these instructions.

Even though I did not see as large of a change as I anticipated, the GPU was still slower executing the conditional kernel than the non conditional kernel.
The CPU was faster for small amounts of kernels, but when I made this number very large the GPU was still able to finish the calculations faster.

![Graph of performance](documentation/branch_timing_chart.png)

Here we can see that the difference between the conditional branching kernel is longer than the non-branching kernel, but not by an exponential amount.
This is why my thoughts above were necessary on this example. If this is not the intended outcome I would love some example code to better replicate the behavior.

## Stretch Problem 




