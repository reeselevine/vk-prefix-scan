#!/bin/bash

# Output file
output_file=$1

runs=$2

# Ensure the output file is empty before starting
echo "" > throughput.dat

# Run the command 2 * $2 times
for i in $(seq 0 $2);
do ./build/"$output_file".run -d 1 -w 32 -t 128 -p 1 -b 'a' >> "throughput.dat"
done
sleep 3
for i in $(seq 0 $2);
do ./build/"$output_file".run -d 1 -w 32 -t 128 -p 1 -b 'a' >> "throughput.dat"
done

python3 benchmark.py $output_file $runs

echo "Completed $runs * 2 runs."
