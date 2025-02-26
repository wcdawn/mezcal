#!/bin/sh

HDF5_VERSION=1.12.1

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx

NPROC='' # default parallel make

cd ${PWD}/../hdf5-${HDF5_VERSION}
CC=${CC} CXX=${CXX} ./configure --enable-cxx || exit 1
CC=${CC} CXX=${CXX} make -j${NPROC} || exit 1
CC=${CC} CXX=${CXX} make install || exit 1
