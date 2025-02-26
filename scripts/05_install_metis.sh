#!/bin/sh

METIS_VERSION=5.1.0
NPROC='' # default parallel compile make

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx

cd ../metis-${METIS_VERSION}
CC=${CC} CXX=${CXX} make BUILDDIR=lib config || exit 1
CC=${CC} CXX=${CXX} make BUILDDIR=lib -j${NPROC} || exit 1
cp lib/libmetis/libmetis.a lib || exit 1
