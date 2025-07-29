#!/bin/sh

set -e
set -u

CU_DIR="cu"
PTX_DIR="ptx"

mkdir -p "$PTX_DIR"

for cu_file in "$CU_DIR"/*.cu; do
    base_name=$(basename "$cu_file" .cu)
    ptx_file="$PTX_DIR/$base_name.ptx"

    echo "Compiling $cu_file -> $ptx_file"
    nvcc -arch=compute_87 -ptx "$cu_file" -o "$ptx_file"
done
