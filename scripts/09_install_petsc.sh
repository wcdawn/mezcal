#!/bin/sh

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx
FC=${PWD}/../mpich-install/bin/mpifort

PETSC_VERSION=3.18.1
PETSC_VERSION=3.20.0
PETSC_DIR=${PWD}/../petsc-${PETSC_VERSION}
PETSC_ARCH=arch-linux-c-debug

NPROC=''

cd ../petsc-${PETSC_VERSION}
./configure CC=${CC} CXX=${CXX} FC=${FC} \
  --with-hypre-dir=../hypre-2.26.0/src/hypre \
  --with-shared-libraries=0 --with-make-np=18 \
  --with-fortran-bindings=0
CC=${CC} CXX=${CXX} FC=${FC} make -j${NPROC} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all
