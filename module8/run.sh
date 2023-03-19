#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 8 assignment"
echo "----------------------------------------------------------------"
echo "This is a provided example runner of how to properly use"
echo "my module 8 assignment."
echo "This assignment demonstrates the curand and cufft libraries"
echo "The cufft example is directly applicable to my proposed project"
echo "where I will be filter audio filter with FIR filters and running"
echo "FFTs on the resulting spectrum"
echo ""

read -p "Press enter to continue:"
clear
echo "Curand example"
echo "Generating 500 uniform random numbers"
nvprof ./assignment_1 500

read -p "Press enter to continue:"
clear
echo "Curand example"
echo "Generating 1000 uniform random numbers"
nvprof ./assignment_1 1000

read -p "Press enter to continue:"
clear
echo "Audio FFT example"
echo "Running FFT with FFT size of 64"
nvprof ./assignment_2 64

read -p "Press enter to continue:"
clear
echo "Audio FFT example"
echo "Running FFT with FFT size of 1024"
nvprof ./assignment_2 1024 
