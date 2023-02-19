#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 4 assignment"
echo "----------------------------------------------------------------"
echo "This runner demonstrates the timing different between paged and"
echo "pinned memory with varying block size and amount of kernels."
echo "The timing is done by nvprof as I found this to be much more"
echo "reliable and accurate than my timing method for module 3."
echo "This runner additionally demonstrates how to run the caesar"
echo "cipher kernel with file I/O."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "Paged memory - 1048576 threads - block size 256"
nvprof ./assignment 1048576 256 0
echo "Pinned memory - 1048576 threads - block size 256"
nvprof ./assignment 1048576 256 1

read -p "Press enter to continue:"
clear
echo "Paged memory - 1048576 threads - block size 128"
nvprof ./assignment 1048576 128 0
echo "Pinned memory - 1048576 threads - block size 128"
nvprof ./assignment 1048576 128 1

read -p "Press enter to continue:"
clear
echo "Paged memory - 16777216 threads - block size 256"
nvprof ./assignment 16777216 256 0
echo "Pinned memory - 16777216 threads - block size 256"
nvprof ./assignment 16777216 256 1

read -p "Press enter to continue:"
clear
echo "Paged memory - 16777216 threads - block size 128"
nvprof ./assignment 16777216 128 0
echo "Pinned memory - 16777216 threads - block size 128"
nvprof ./assignment 16777216 128 1

read -p "Press enter to continue"
clear
echo "Caesar Cipher example"
echo "Input romeo_and_juliet.txt - block size - 256"
echo "output - romeo_and_juliet_encrypted.txt - rot = 13"
./caesar_cipher 256 romeo_and_juliet.txt romeo_and_juliet_encrypted.txt 13
