#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 9 thrust assignment"
echo "----------------------------------------------------------------"
echo "This runner demonstrates the add, sub, mult, and mod operations"
echo "with thrust with varying vector sizes."
echo "This runner also compares with my module 3 assignment which uses"
echo "the same kernel operations."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "Thrust addition: running with vector size of 10"
nvprof ./thrust_assignment 10 0

read -p "Press enter to continue:"
clear
echo "Thrust subtraction: running with vector size of 10"
nvprof ./thrust_assignment 10 1

read -p "Press enter to continue:"
clear
echo "Thrust multiplication: running with vector size of 10"
nvprof ./thrust_assignment 10 2

read -p "Press enter to continue:"
clear
echo "Thrust modulo: running with vector size of 10"
nvprof ./thrust_assignment 10 3

read -p "Press enter to continue:"
clear
echo "Thrust addition: running with vector size of 10000"
nvprof ./thrust_assignment 10000 0

read -p "Press enter to continue:"
clear
echo "Thrust subtraction: running with vector size of 10000"
nvprof ./thrust_assignment 10000 1

read -p "Press enter to continue:"
clear
echo "Thrust multiplication: running with vector size of 10000"
nvprof ./thrust_assignment 10000 2

read -p "Press enter to continue:"
clear
echo "Thrust modulo: running with vector size of 10000"
nvprof ./thrust_assignment 10000 3

read -p "Press enter to continue:"
clear
echo "Module 3 addiition: running with 10 threads - block size of 10"
nvprof ./module3_assignment 10 10 0

read -p "Press enter to continue:"
clear
echo "Module 3 subtraction: running with 10 threads - block size of 10"
nvprof ./module3_assignment 10 10 1

read -p "Press enter to continue:"
clear
echo "Module 3 multiplication: running with 10 threads - block size of 10"
nvprof ./module3_assignment 10 10 2

read -p "Press enter to continue:"
clear
echo "Module 3 modulo: running with 10 threads - block size of 10"
nvprof ./module3_assignment 10 10 3

read -p "Press enter to continue:"
clear
echo "Module 3 addiition: running with 10000 threads - block size of 100"
nvprof ./module3_assignment 10000 100 0

read -p "Press enter to continue:"
clear
echo "Module 3 subtraction: running with 10000 threads - block size of 100"
nvprof ./module3_assignment 10000 100 1

read -p "Press enter to continue:"
clear
echo "Module 3 multiplication: running with 10000 threads - block size of 100"
nvprof ./module3_assignment 10000 100 2

read -p "Press enter to continue:"
clear
echo "Module 3 modulo: running with 10000 threads - block size of 100"
nvprof ./module3_assignment 10000 100 3
