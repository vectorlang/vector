#!/bin/bash

action=link

while getopts "scgd" opt
do
    case "$opt" in
        s) action=generate ;;
        c) action=compile ;;
        g) debug="debug" ;;
        d) device="device" ;;
    esac
done

shift $((OPTIND-1))

if [ "$debug" ]; then
    CFLAGS="-Wall -g"
else
    CFLAGS="-Wall -O2"
fi

if [ "$device" ]; then
    NVCC_FLAGS="-arch=sm_11 -I./rtlib -Xcompiler $CFLAGS -DBLOCK_SIZE=256"
    LD_FLAGS="-L/opt/cuda/lib64 -lcudart -lm"
else
    NVCC_FLAGS="-arch=sm_20 -I./rtlib -Xcompiler $CFLAGS"
    LD_FLAGS="-L/usr/local/cuda/lib -locelot -lm"
fi

if [ "$debug" ]; then
    LD_FLAGS="$LD_FLAGS -g"
fi

fname=$1
fname_noext="${fname%.*}"

echo "./compiler/generator < $fname > ${fname_noext}.cu"
./compiler/generator < "$fname" > "${fname_noext}.cu"

if [ "$action" != "generate" ]; then
    echo "nvcc -c $NVCC_FLAGS ${fname_noext}.cu -o ${fname_noext}.o"
    nvcc -c $NVCC_FLAGS "${fname_noext}.cu" -o "${fname_noext}.o"
fi

if [ "$action" == "link" ]; then
    echo "g++ $LD_FLAGS ${fname_noext}.o -o ${fname_noext}"
    g++ $LD_FLAGS "${fname_noext}.o" -o "${fname_noext}"
fi

if [ "$action" != "generate" ]; then
    rm -f "${fname_noext}.cu"
fi

if [ "$action" == "link" ]; then
    rm -f "${fname_noext}.o"
fi
