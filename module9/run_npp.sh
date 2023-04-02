#!/bin/bash

echo "Frederich Stine - EN 605.617 - JHU EP"
echo "----------------------------------------------------------------"
echo "Example runner to show execution of Module 9 npp assignment"
echo "----------------------------------------------------------------"
echo "This runner runs my example npp code that applies a gaussian"
echo "blur to a grayscale image."
echo "This runner runs through two separate grayscale images differing in"
echo "size by 512x512 to 5184x3456"

read -p "Press enter to continue:"
clear
echo "NPP gaussian blur: file of 512x512"
nvprof ./npp_assignment images/Lena.pgm images/Lena-gauss.pgm

read -p "Press enter to continue:"
clear
echo "NPP gaussian blur: file of 5184x3456"
nvprof ./npp_assignment images/sample.pgm images/sample-gauss.pgm 
