#!/bin/sh

# TODO revisit with bigint

CUDA_HOME=${OLCF_CUDA_ROOT} \
  HYPRE_CUDA_SM=70 \
  CC=mpixlc \
  CXX=mpixlC \
  ./configure \
  --disable-fortran \
  --with-cuda \
  --enable-unified-memory \
  --enable-device-memory-pool \
  --enable-cusparse \
  --enable-cublas \
  --enable-curand

CC=mpixlc
  CXX=mpixlC \
  ./configure \
  --disable-fortran
