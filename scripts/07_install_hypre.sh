#!/bin/sh

HYPRE_VERSION=2.26.0
NPROC='' # default parallel make

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx

cd ../hypre-${HYPRE_VERSION}/src
CC=${CC} CXX=${CXX} ./configure --disable-fortran
CC=${CC} CXX=${CXX} make -j${NPROC}
