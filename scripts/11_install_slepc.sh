#!/bin/sh

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx

PETSC_VERSION=3.18.1
PETSC_VERSION=3.20.0
PETSC_DIR=${PWD}/../petsc-${PETSC_VERSION}
PETSC_ARCH=arch-linux-c-debug
SLEPC_DIR=${PWD}/../slepc-${PETSC_VERSION}

NPROC=''

cd ../slepc-${PETSC_VERSION}
CC=${CC} CXX=${CXX} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} ./configure
CC=${CC} CXX=${CXX} make -j${NPROC} PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} SLEPC_DIR=${SLEPC_DIR} all
