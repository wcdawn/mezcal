#!/bin/sh

# TODO revisit with bigint

CUDA_HOME=/opt/cuda \
  HYPRE_CUDA_SM=75 \
  CC=mpicc \
  CXX=mpicxx \
  ./configure \
  --disable-fortran \
  --with-cuda \
  --enable-cusparse \
  --enable-cublas \
  --enable-curand \
  --enable-unified-memory \
  --enable-device-memory-pool
