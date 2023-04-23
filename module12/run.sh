#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 12 assignment"
echo "----------------------------------------------------------------"
echo " This runner demonstrates a an averaging filter using 4x1 sub-buffers."
echo " The input signal is known."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "OpenCL sub-buffer"
echo "Run 1"
./assignment

read -p "Press enter to continue:"
clear
echo "OpenCL sub-buffer"
echo "Run 2"
./assignment

read -p "Press enter to continue:"
clear
echo "OpenCL sub-buffer"
echo "Run 3"
./assignment