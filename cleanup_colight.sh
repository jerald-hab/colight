#!/bin/bash

echo "=== Killing runexp processes ==="

# Kill any python processes running runexp*.py
PIDS=$(ps -ef | grep "python -O runexp" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No runexp processes found."
else
    echo "Killing PIDs: $PIDS"
    kill -9 $PIDS
fi

echo "=== Removing training logs ==="
rm -f training6.log

#echo "=== Removing model checkpoints ==="
#rm -rf model/*Colight*

echo "=== Removing records ==="
rm -rf records/*

echo "=== Cleanup complete ==="
