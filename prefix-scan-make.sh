#!/bin/bash

set -e  # Stop script on any error

BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512 1024 2048)

for BATCH in "${BATCH_SIZES[@]}"; do    
        sed -i "s/#define BATCH_SIZE.*/#define BATCH_SIZE $BATCH/" prefix-scan.cl
        make prefix-scan

        if [ -f "build/prefix-scan.cinit" ]; then
            echo "Moving: build/prefix-scan.cinit -> /batch_size/prefix-scan${BATCH}_uint4.cinit"
            mv -f "build/prefix-scan.cinit" "batch_size/prefix-scan${BATCH}_uint4.cinit"
            ls -l "batch_size/prefix-scan${BATCH}_uint4.cinit"
        else
            echo "Error: build/prefix-scan.cinit not found!"
            exit 1
        fi

        sed -i 's/__global uint4/__global uint2/g' prefix-scan.cl
        sed -i 's/#define U4/#define U2/g' prefix-scan.cl

        make prefix-scan
        if [ -f "build/prefix-scan.cinit" ]; then
            echo "Moving: build/prefix-scan.cinit -> /batch_size/prefix-scan${BATCH}_uint2.cinit"
            mv -f "build/prefix-scan.cinit" "batch_size/prefix-scan${BATCH}_uint2.cinit"
            ls -l "batch_size/prefix-scan${BATCH}_uint2.cinit"
        else
            echo "Error: build/prefix-scan.cinit not found!"
            exit 1
        fi

        sed -i 's/__global uint2/__global uint4/g' prefix-scan.cl
        sed -i 's/#define U2/#define U4/g' prefix-scan.cl

done