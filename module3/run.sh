#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 3 assignment"
echo "----------------------------------------------------------------"
echo "This runner demonstrates the add, sub, mult, and mod operations"
echo "with varying block sizes and thread counts."
echo "This runner additionally demonstrates conditional branching both"
echo "in a Cuda kernel and outside with a reference Cuda kernel that"
echo "does not include conditional branching."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "Portion 1 - Cuda addition kernel"
echo "Running with 128 threads - block size of 16"
./assignment 128 16 0
read -p "Press enter to continue:"
clear
echo "Running with 128 threads - block size of 32"
./assignment 128 32 0
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 16"
./assignment 256 16 0
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 32"
./assignment 256 32 0

read -p "Press enter to continue:"
clear
echo "Portion 2 - Cuda subtraction kernel"
echo "Running with 128 threads - block size of 16"
./assignment 128 16 1
read -p "Press enter to continue:"
clear
echo "Running with 128 threads - block size of 32"
./assignment 128 32 1
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 16"
./assignment 256 16 1
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 32"
./assignment 256 32 1

read -p "Press enter to continue:"
clear
echo "Portion 3 - Cuda multiplication kernel"
echo "Running with 128 threads - block size of 16"
./assignment 128 16 2
read -p "Press enter to continue:"
clear
echo "Running with 128 threads - block size of 32"
./assignment 128 32 2
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 16"
./assignment 256 16 2
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 32"
./assignment 256 32 2

read -p "Press enter to continue:"
clear
echo "Portion 4 - Cuda modulo kernel"
echo "Running with 128 threads - block size of 16"
./assignment 128 16 3
read -p "Press enter to continue:"
clear
echo "Running with 128 threads - block size of 32"
./assignment 128 32 3
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 16"
./assignment 256 16 3
read -p "Press enter to continue:"
clear
echo "Running with 256 threads - block size of 32"
./assignment 256 32 3

read -p "Press enter to continue:"
clear
echo "Portion 5 - Cuda conditional branch timing test"
echo "Running with 1024 threads - block size of 64"
./assignment 1024 64 4
echo "Running with 2048 threads - block size of 64"
./assignment 2048 64 4
echo "Running with 4096 threads - block size of 64"
./assignment 4096 64 4
echo "Running with 1024 threads - block size of 128"
./assignment 1024 128 4
echo "Running with 2048 threads - block size of 128"
./assignment 2048 128 4
echo "Running with 4096 threads - block size of 128"
./assignment 4096 128 4

