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
echo "These assignment runs with a fixed quantity of threads"
echo "at 1024. There are two varying block sizes in this runner."
echo ""

read -p "Press enter to continue:"
clear
echo "Portion 1 - Copy to shared memory"
echo "Running with block size of 128 -"
nvprof ./assignment 128 0

read -p "Press enter to continue:"
clear
echo "Portion 2 - Shared memory for local"
echo "Running with block size of 128 -"
nvprof ./assignment 128 1

read -p "Press enter to continue:"
clear
echo "Portion 3 - Copy to constant memory"
echo "Running with block size of 128 -"
nvprof ./assignment 128 2

read -p "Press enter to continue:"
clear
echo "Portion 4 - Constant memory only"
echo "Running with block size of 128 -"
nvprof ./assignment 128 3

read -p "Press enter to continue:"
clear
echo "Portion 5 - Copy to shared memory"
echo "Running with block size of 256 -"
nvprof ./assignment 256 0

read -p "Press enter to continue:"
clear
echo "Portion 6 - Shared memory for local"
echo "Running with block size of 256 -"
nvprof ./assignment 256 1

read -p "Press enter to continue:"
clear
echo "Portion 7 - Copy to constant memory"
echo "Running with block size of 256 -"
nvprof ./assignment 256 2

read -p "Press enter to continue:"
clear
echo "Portion 8 - Constant memory only"
echo "Running with block size of 256 -"
nvprof ./assignment 256 3
