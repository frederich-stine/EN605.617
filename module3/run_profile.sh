#!/bin/bash

echo "Run conditional code to generate graph"

THREADS=128

while [ $THREADS -le 120000 ]
do
	echo ./assignment $THREADS 128 4
	./assignment $THREADS 128 4
	THREADS=$(($THREADS*2))
done


