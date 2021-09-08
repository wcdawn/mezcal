#!/bin/sh

PETSC_DIR=/home/wcdawn/projects/container/petsc
  PETSC_ARCH=arch-linux-c-opt \
  SLEPC_DIR=/home/wcdawn/projects/container/slepc \
  CC=mpicc \
  CXX=mpicxx \
  ./configure
