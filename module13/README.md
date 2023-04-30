# Module 13

This lab created a new OpenCL program that created a different number of events based on the input from a file passed to the program.
In order to create inputs for the program I created a Python program called *generate_rand.py*. This program outputs floating point uniform random variables from 0-10 to a text file with a specified number of elements per line and number of lines.
This data is read into the assignment with a custom parser, this time is not included in the timing information for this assignment.
In order to create multiple events this assignment creates multiple kernels (one for each line in the input).
There are events created for each of these kernels. After running the first kernels (square) the program then launches kernels to average the data.
This averaging depends on the execution of the first events so the program waits for this arbitrary number of events to finish.
Events are used again to track the finish of the averaging kernels.

The *generate_rand.py* program utilizes command line arguments. These can be seen below:

```bash
frederich@weak:~/Documents/EN605.617/module13$ python3 generate_rand.py 
usage: Rand Generator [-h] -f FILE -c COUNT -w WIDTH
Rand Generator: error: the following arguments are required: -f/--file, -c/--count, -w/--width
```

The *assignment* program also utilizes command line arguments to take the name of an input file.

```bash
frederich@weak:~/Documents/EN605.617/module13$ ./assignment 
Error: Not enough arguments
Correct usage is:
    : ./assignment {filename}
```

For timing I used std::chrono starting from the dispatch of kernels to the end of execution of the events. I did this
for both kernels executed. This was more straightforward than my previous technique of using profiling events as this
program creates a lot of events that represent kernels with very short runtimes.  

I additionally created a *run.sh* script that runs six executions of the program. The input values come from a pre-generated set of inputs in the [inputs](inputs) folder. A log of this *run.sh* script is available in [documentation/full_run.log](documentation/full_run.log).  

Solely the timing informtaion from this log is shown below:

```bash
Input 16x16 Square and Average
Count: 16.000000
Width: 16.000000

Square kernel runtime: 0.000137s
Average kernel runtime: 0.000137s

Input 128x128 Square and Average
Count: 128.000000
Width: 128.000000

Square kernel runtime: 0.000546s
Average kernel runtime: 0.002558s

Input 256x256 Square and Average
Count: 256.000000
Width: 256.000000

Square kernel runtime: 0.000976s
Average kernel runtime: 0.008898s

Input 1024x1024 Square and Average
Count: 1024.000000
Width: 1024.000000

Square kernel runtime: 0.004174s
Average kernel runtime: 0.147698s

Input 64x4096 Square and Average
Count: 64.000000
Width: 4096.000000

Square kernel runtime: 0.000244s
Average kernel runtime: 0.037611s

Input 4096x64 Square and Average
Count: 4096.000000
Width: 64.000000

Square kernel runtime: 0.019188s
Average kernel runtime: 0.046601s
```

Here we can see that the square kernel (although it reaches the same amount of data) takes less time. This is due to
how the kernels work, the second averaging kernel runs with one thread across the whole array of input. After doing
some research I found that you can do a more efficient reduction by averaging two values with each kernel slowly going
down to the final two values. This was more for demonstration purposes of creating a kernel that depended on the results
of the previous kernel rather than a highly efficient algorithm.

This assignment taught me about events in OpenCL. I had been using events previously but this showed me a few new ways to use them.
This assignment also confused me quite a bit. First I thought that the assignment was trying to create an program that
had conditional behavior based on OpenCL events. It does not look like this is an intended use case of events.
I thought about trying to use event callback functions, but this still did not make sense. I did the best that I could
come up with an am happy with the resutls that I got.
