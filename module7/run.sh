#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 7 assignment"
echo "----------------------------------------------------------------"
echo "This is a provided example runner of how to properly use"
echo "my module 7 assignment."
echo "This assignment runs through an arith kernel from module 6"
echo "using both default and streamed execution. The amount of"
echo "streams is variable and is changed throughout this run.sh script"
echo "    0: Streamed execution"
echo "    1: Synchronous execution"
echo ""

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 1024 threads with block size of 64 - 2 streams"
nvprof ./assignment 1024 64 0 2

read -p "Press enter to continue:"
clear
echo "1: Synchronous execution"
echo "Running 1024 threads with block size of 64"
nvprof ./assignment 1024 64 1 

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 64 - 2 streams"
nvprof ./assignment 64000 64 0 2

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 64 - 10 streams"
nvprof ./assignment 64000 64 0 10

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 64 - 100 streams"
nvprof ./assignment 64000 64 0 100

read -p "Press enter to continue:"
clear
echo "0: Synchronous execution"
echo "Running 64000 threads with block size of 64"
nvprof ./assignment 64000 64 1

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 128 - 2 streams"
nvprof ./assignment 64000 128 0 2

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 128 - 10 streams"
nvprof ./assignment 64000 128 0 10

read -p "Press enter to continue:"
clear
echo "0: Streamed execution"
echo "Running 64000 threads with block size of 128 - 100 streams"
nvprof ./assignment 64000 128 0 100

read -p "Press enter to continue:"
clear
echo "0: Synchronous execution"
echo "Running 64000 threads with block size of 128"
nvprof ./assignment 64000 128 1

