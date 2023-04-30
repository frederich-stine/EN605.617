#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 13 assignment"
echo "----------------------------------------------------------------"
echo " This runner demonstrates a differing events, and kernel execution."
echo " Based on an input file provided."
echo " The input file should be created by generate_rand.py for this program."
echo ""
echo ""

read -p "Press enter to continue:"
clear
echo "Input 16x16 Square and Average"
./assignment inputs/input_16x16.txt

read -p "Press enter to continue:"
clear
echo "Input 128x128 Square and Average"
./assignment inputs/input_128x128.txt

read -p "Press enter to continue:"
clear
echo "Input 256x256 Square and Average"
./assignment inputs/input_256x256.txt

read -p "Press enter to continue:"
clear
echo "Input 256x256 Square and Average"
./assignment inputs/input_256x256.txt

read -p "Press enter to continue:"
clear
echo "Input 1024x1024 Square and Average"
./assignment inputs/input_1024x1024.txt

read -p "Press enter to continue:"
clear
echo "Input 64x4096 Square and Average"
./assignment inputs/input_64x4096.txt

read -p "Press enter to continue:"
clear
echo "Input 4096x64 Square and Average"
./assignment inputs/input_4096x64.txt
