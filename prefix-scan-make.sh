#!/bin/bash

set -e  # Stop script on any error

BATCH_SIZES=(1 2 4 8)

for BATCH in "${BATCH_SIZES[@]}"; do
    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE $BATCH/" prefix-scan.cl

    make prefix-scan

    if [ -f "build/prefix-scan.cinit" ]; then
        echo "Moving: build/prefix-scan.cinit -> /batch_size/prefix-scan${BATCH}.cinit"
        mv -f "build/prefix-scan.cinit" "batch_size/prefix-scan${BATCH}.cinit"
        ls -l "batch_size/prefix-scan${BATCH}.cinit"
    else
        echo "Error: build/prefix-scan.cinit not found!"
        exit 1
    fi
done