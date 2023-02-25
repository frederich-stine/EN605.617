# Module 6 - Frederich Stine

## Registers

To demonstrate the speed difference of registers versus global memory I created three separate kernels

- One that copies all data used into registers
- One that uses registers as a local variable
- One that purely uses global memory

These kernels, differently from the previous assignment, perform operations in a loop of 10000 iterations inside of the kernel.
With the last assignment I found that it was hard to see the real difference between memory speedup due to how fast 
the kernels executed. This made the relationship very easy to see, but I think thta the previous assignment is 
still a valid testing case as the launch and copy time of the kernel will likely always be a large portion of the 
execution time.

This program allows for a variable size of threads and block size. The most register intensive kernel only utilizes 4 
registers, so a lack of registers on the device should not be an issue.

As with previous assignments, all functionality is packed into one *assignment.cu* file. Different functions are invoked 
from the command line arguments. The help menu is below:

```
Call ./assignment {threadCount} {blockSize} {operation}
Operations: 
    0: Copy to register
    1: Register local variables
    2: Global memory only
```

As per usual I created a *run.sh* script that uses nvprof to time different elements of the program. The output of some 
of the execution of this script is shown below. The full log is available in [documentation/full_run.log](documentation/full_run.log).

```

```
