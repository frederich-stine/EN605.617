# Module 11

This lab modified an OpenCL program that performed a convolution. I expanded the original program to work with floating 
point values instead of integers. I also changed the program to work with 49x49 signals and a set 7x7 convolution. This 
assignment is in *assignment.cpp* and the kernel is in *assignment.cl*.

I used the same profiling event that I used with the module 10 assignment to measure the execution time of the kernel.

There are no arguments passed to this assignment.

I additionally created a *run.sh* script that runs three executions of the program. Since the values for the input 
are generated randomly this results in the kernel being run with different input data. A log of this *run.sh* scrip is 
available in [documentation/full_run.log](documentation/full_run.log).

Here is just the timing information from the log.
```
OpenCl Execution time is: 0.024 milliseconds 
OpenCl Execution time is: 0.026 milliseconds 
OpenCl Execution time is: 0.024 milliseconds 
```

From this timing information we can see that the kernel execution time varies very slighly. This is likely due to variance 
on the GPU and is likely not a result of a different from the input data. Since the same operations are done no matter what 
data is passed, the differernce in input data should not create any sort of timing difference.

This assignment was helpful in learning how to perform convolutions in OpenCL. This is an operation that is prolific 
in signal processing and machine learning.
