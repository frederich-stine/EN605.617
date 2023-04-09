#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 10 assignment"
echo "----------------------------------------------------------------"
echo "This runner demonstrates the add, sub, mult, mod, and pow operations"
echo "with varying array sizes in OpenCL."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "OpenCL addition kernel"
echo "Running with 100 arrSize"
./assignment 100 0

read -p "Press enter to continue:"
clear
echo "OpenCL subtraction kernel"
echo "Running with 100 arrSize"
./assignment 100 1

read -p "Press enter to continue:"
clear
echo "OpenCL multiplication kernel"
echo "Running with 100 arrSize"
./assignment 100 2

read -p "Press enter to continue:"
clear
echo "OpenCL division kernel"
echo "Running with 100 arrSize"
./assignment 100 3

read -p "Press enter to continue:"
clear
echo "OpenCL pow kernel"
echo "Running with 100 arrSize"
./assignment 100 4

read -p "Press enter to continue:"
clear
echo "OpenCL addition kernel"
echo "Running with 100000 arrSize"
./assignment 100000 0

read -p "Press enter to continue:"
clear
echo "OpenCL subtraction kernel"
echo "Running with 100000 arrSize"
./assignment 100000 1

read -p "Press enter to continue:"
clear
echo "OpenCL multiplication kernel"
echo "Running with 100000 arrSize"
./assignment 100000 2

read -p "Press enter to continue:"
clear
echo "OpenCL division kernel"
echo "Running with 100000 arrSize"
./assignment 100000 3

read -p "Press enter to continue:"
clear
echo "OpenCL pow kernel"
echo "Running with 100000 arrSize"
./assignment 100000 4
