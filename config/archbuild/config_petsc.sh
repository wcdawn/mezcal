#!/bin/sh

# TODO figure out hypre...
  #--with-hypre-dir=/ccs/proj/csc404/wcdawn/container/hypre-2.22.0/src/hypre \

# TODO revisit when hypre works with 64 bit integers
# petsc & hypre must use same size integers
  #--with-64-bit-indices=1 \

./configure \
  --COPTFLAGS='-O3' \
  --CXXOPTFLAGS='-O3' \
  --CUDAOPTFLAGS='-O3' \
  --with-cc=mpicc \
  --with-cxx=mpicxx \
  --with-fc=0 \
  --with-debugging=no \
  --download-f2cblaslapack=1 \
  --with-make-np=24 \
  PETSC_ARCH=arch-linux-c-opt \
  PETSC_DIR=/home/wcdawn/projects/container/petsc
