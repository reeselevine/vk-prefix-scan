#!/bin/bash

set -e  # Stop script on any error

BATCH_SIZES=(1 2 4 8)

for BATCH in "${BATCH_SIZES[@]}"; do
    sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE $BATCH/" blit.cl

    make blit

    if [ -f "build/blit.cinit" ]; then
        echo "Moving: build/blit.cinit -> /batch_size/blit${BATCH}.cinit"
        mv -f "build/blit.cinit" "batch_size/blit${BATCH}.cinit"
        ls -l "batch_size/blit${BATCH}.cinit"
    else
        echo "Error: build/blit.cinit not found!"
        exit 1
    fi
done