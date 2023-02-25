#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 5 assignment"
echo "----------------------------------------------------------------"
echo "This is a provided example runner of how to properly use"
echo "my module 5 assignment."
echo "This assignment runs through 4 different kernels:"
echo "    0: Arith with full copy to shared memory"
echo "    1: Arith with shared memory for local variables"
echo "    2: Arith with copy to constant memory"
echo "    3: Arith with constant memory only"
echo ""

read -p "Press enter to continue:"
clear
echo "Portion 1 - Copy to registers"
echo "Running 1024 threads with block size of 128 -"
nvprof ./assignment 1024 128 0

read -p "Press enter to continue:"
clear
echo "Portion 2 - Register and global"
echo "Running 1024 threads with block size of 128 -"
nvprof ./assignment 1024 128 1

read -p "Press enter to continue:"
clear
echo "Portion 3 - Global memory only"
echo "Running 1024 threads with block size of 128 -"
nvprof ./assignment 1024 128 2

read -p "Press enter to continue:"
clear
echo "Portion 1 - Copy to registers"
echo "Running 1024 threads with block size of 256 -"
nvprof ./assignment 1024 256 0

read -p "Press enter to continue:"
clear
echo "Portion 2 - Register and global"
echo "Running 1024 threads with block size of 256 -"
nvprof ./assignment 1024 256 1

read -p "Press enter to continue:"
clear
echo "Portion 3 - Global memory only"
echo "Running 1024 threads with block size of 256 -"
nvprof ./assignment 1024 256 2

read -p "Press enter to continue:"
clear
echo "Portion 1 - Copy to registers"
echo "Running 1000000 threads with block size of 256 -"
nvprof ./assignment 1000000 256 0

read -p "Press enter to continue:"
clear
echo "Portion 2 - Register and global"
echo "Running 1000000 threads with block size of 256 -"
nvprof ./assignment 1000000 256 1

read -p "Press enter to continue:"
clear
echo "Portion 3 - Global memory only"
echo "Running 1000000 threads with block size of 256 -"
nvprof ./assignment 1000000 256 2
