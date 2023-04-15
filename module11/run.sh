#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 11 assignment"
echo "----------------------------------------------------------------"
echo " This runner demonstrates a convolution with a mask of 7x7 on a"
echo " signal of 49x49. The input signal is randomly generated showing"
echo " the effects of different values on the performance of the kernel"
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "OpenCL convolution kernel"
echo "Run 1"
./assignment

read -p "Press enter to continue:"
clear
echo "OpenCL convolution kernel"
echo "Run 2"
./assignment

read -p "Press enter to continue:"
clear
echo "OpenCL convolution kernel"
echo "Run 3"
./assignment
