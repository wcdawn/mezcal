#!/bin/sh

PETSC_DIR=/ccs/proj/csc404/wcdawn/container/petsc \
  PETSC_ARCH=arch-olcf-summit-opt \
  SLEPC_DIR=/ccs/proj/csc404/wcdawn/container/slepc \
  CC=mpixlc \
  CXX=mpixlC \
  ./configure
