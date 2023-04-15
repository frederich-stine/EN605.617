# Module 10

This lab was the first lab that utilized OpenCL. I wrote 5 different arithmatic kernels in OpenCL to expand on the 
*HelloWorld* design. I was suprised by how similar writing kernels in OpenCL is to writing kernels in Cuda. This 
assignment is in *assignment.cpp* and the kernels are in *assignment.cl*.

I followed a guide on how to utilize OpenCL profiling events to determine the quantity of time that my code spend 
executing the arithmetic kernels. This was a very nice feature provided by OpenCL to allow for timing of GPU functions.
https://stackoverflow.com/questions/23550912/measuring-execution-time-of-opencl-kernels

I also modified the *HelloWorld* design such that the arrSize is variable based on user input. This is statically 
allocated to the maximum size is SIZE_MAX.

The arguments passed to the assignment are:
```
Call ./assignment {arrSize} {operation}
Operations: 
    0: Add
    1: Sub
    2: Mult
    3: Div
    4: Pow
```

I additionally created a *run.sh* script that runs two executions of each kernel and outputs timing details. 
I did not output the results of the computation when the array size gets over 1000 to help with readability. This 
log is available in [documentation/full_run.log](documentation/full_run.log).

Here is the output from the log to discuss the timing characteristics.
```
Frederich Stine - EN 605.617 - JHU EP
----------------------------------------------------------------
Example runner to show execution of Module 10 assignment
----------------------------------------------------------------
This runner demonstrates the add, sub, mult, mod, and pow operations
with varying array sizes in OpenCL.


Press enter to continue:
OpenCL addition kernel
Running with 100 arrSize
8 2 31 16 27 23 15 20 26 30 15 25 20 15 20 16 26 11 16 31 15 25 3 8 16 34 7 17 18 1 16 10 20 17 37 26 8 20 4 17 1 16 27 14 13 19 17 36 28 11 30 31 24 26 17 13 11 16 11 1 18 26 27 21 25 19 6 11 19 15 9 19 15 14 10 19 25 20 17 13 7 26 16 20 35 28 2 20 27 24 27 12 7 16 7 15 8 5 15 8 
Executed program succesfully.
OpenCl Execution time is: 0.008 milliseconds 
Press enter to continue:
OpenCL subtraction kernel
Running with 100 arrSize
0 -4 14 10 -2 3 16 3 6 11 -2 -12 9 -1 -13 -12 -2 -2 10 14 8 -3 -3 9 -9 0 -4 6 -8 -5 3 5 7 -8 2 -11 0 5 -5 -12 14 6 4 -14 6 -6 -5 -10 10 -10 5 -6 13 -17 14 -2 7 -3 -2 0 -17 -7 1 9 -8 -13 10 15 -10 4 -12 0 6 0 9 8 13 11 3 -10 4 -7 -5 -14 -16 9 10 12 6 -8 3 -1 12 -9 -10 -9 6 4 -9 -2 
Executed program succesfully.
OpenCl Execution time is: 0.009 milliseconds 
Press enter to continue:
OpenCL multiplication kernel
Running with 100 arrSize
16 285 32 96 288 70 36 54 91 102 99 64 162 42 0 64 99 288 56 32 84 4 88 10 190 361 165 7 65 0 304 104 8 84 15 152 81 150 204 45 51 0 252 15 7 0 126 119 75 39 24 40 0 0 51 168 198 54 35 16 0 30 6 190 105 48 144 54 0 117 85 256 16 25 10 20 90 12 28 56 77 18 84 95 36 0 75 64 7 153 154 156 64 90 0 22 40 117 162 99 
Executed program succesfully.
OpenCl Execution time is: 0.009 milliseconds 
Press enter to continue:
OpenCL division kernel
Running with 100 arrSize
18 0 3.16667 1 4.5 1.77778 0.7 3 0.526316 inf 0.714286 0.166667 2.25 0.157895 0.866667 0.166667 8 1.2 0.5 0 1.25 0.555556 0.625 0.923077 2.66667 0.722222 0 1 1.08333 0.166667 2.33333 6 1.88889 0.588235 2.5 1.2 0.222222 0.416667 1.88889 1 1.45455 2 3.8 0.833333 1.54545 4.75 0.666667 6.33333 18 0.8 1.5 0.176471 1.375 1 1.42857 0.333333 1.28571 0.352941 1.1875 0.25 7 0.307692 0.666667 8 inf 1.45455 0.157895 0 0 0.7 0.166667 0.421053 0 0.666667 0.3 0 0.6 1.33333 0.75 0.777778 3 3 0.6 3.33333 0.388889 0.625 1.23077 1.1875 2.71429 0 10 1.30769 3.6 1.1 0.0555556 5.5 0.125 0.352941 0.111111 inf 
Executed program succesfully.
OpenCl Execution time is: 0.010 milliseconds 
Press enter to continue:
OpenCL pow kernel
Running with 100 arrSize
18 0 4.70459e+07 3125 81 6.87195e+10 2.82475e+08 36 1e+19 1 1e+14 3.87421e+08 6561 1.16226e+09 5.11859e+16 4096 256 6.74664e+18 2.81475e+14 0 1e+08 1e+18 1e+16 1.06993e+14 1.67772e+07 1.12455e+20 0 3.87421e+08 2.32981e+13 1 343 5832 1.18588e+11 1e+17 25 6.74664e+18 512 2.44141e+08 1.18588e+11 3125 1.75922e+13 4.29497e+09 2.4761e+06 1e+12 3.42719e+13 130321 1.00777e+07 6859 18 1.5407e+16 5.7665e+11 1.2914e+08 2.14359e+08 1 1e+07 1.0156e+14 3.74813e+17 1.69267e+13 2.88441e+20 256 7 6.71089e+07 1e+15 8 1 1.75922e+13 1.16226e+09 0 0 2.82475e+08 1 1.44115e+17 0 6.87195e+10 59049 0 243 262144 1.84884e+17 4.03536e+07 729 20736 243 1000 1.62841e+15 390625 4.5036e+15 2.88441e+20 8.93872e+08 0 10 9.90458e+15 1.88957e+06 2.59374e+10 1 121 65536 1.69267e+13 1 1 
Executed program succesfully.
OpenCl Execution time is: 0.009 milliseconds 
Press enter to continue:
OpenCL addition kernel
Running with 100000 arrSize
Executed program succesfully.
OpenCl Execution time is: 0.139 milliseconds 
Press enter to continue:
OpenCL subtraction kernel
Running with 100000 arrSize
Executed program succesfully.
OpenCl Execution time is: 0.140 milliseconds 
Press enter to continue:
OpenCL multiplication kernel
Running with 100000 arrSize
Executed program succesfully.
OpenCl Execution time is: 0.140 milliseconds 
Press enter to continue:
OpenCL division kernel
Running with 100000 arrSize
Executed program succesfully.
OpenCl Execution time is: 0.141 milliseconds 
Press enter to continue:
OpenCL pow kernel
Running with 100000 arrSize
Executed program succesfully.
OpenCl Execution time is: 0.141 milliseconds 
```

From this outtput we can see that the different kernels have almost the same execution time for a given array size. 
This shows that the computation happening inside of the kernels does not vary enough to create a large variance in 
computation time.  We can also see that this time does scale up with the size of the array. With an array of 100000 
we see an increase of around 15.5x. This is considerably less than the 1000x increase in array size. This shows 
that even though I am only timing the execution of the kernel, startup of the kernel still plays a large percentage 
of the execution time.

I'm excited to get with OpenCL more and like that it works on more platforms than Cuda. In the future I'm interested 
to see how well knowledge from Cuda will transfer over to OpenCL.
