#!/bin/sh

MPICH_VERSION=4.0.3
NPROC=''

PWD=$(pwd)
cd ${PWD}/../mpich-${MPICH_VERSION}
FFLAGS=-fallow-argument-mismatch FCFLAGS=-fallow-argument-mismatch ./configure --prefix=${PWD}/../mpich-install 2>&1 | tee c.txt || exit 1
make -j${NPROC} 2>&1 | tee m.txt || exit 1
make install 2>&1 | tee mi.txt || exit 1
