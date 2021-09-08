#!/bin/sh

make config \
  MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES CC=mpixlc CXX=mpixlC \
  MPICC=mpixlc MPICXX=mpixlC \
  HYPRE_DIR=/ccs/proj/csc404/wcdawn/container/hypre/src/hypre \
  METIS_DIR=/ccs/proj/csc404/wcdawn/container/metis-5.1.0 MFEM_USE_METIS_5=YES \
  MFEM_USE_CUDA=YES CUDA_CXX=nvcc \
  CUDA_FLAGS='-x=cu -arch=sm_70 --expt-extended-lambda --compiler-options -O3' \
  CUDA_LIB='-lcusparse -lcurand -lcublas' \
  MFEM_USE_AMGX=YES \
  MFEM_USE_PETSC=YES \
  PETSC_DIR=/ccs/proj/csc404/wcdawn/container/petsc/arch-olcf-summit-opt \
  PETSC_ARCH=arch-olcf-summit-opt \
  MFEM_USE_SLEPC=YES \
  SLEPC_DIR=/ccs/proj/csc404/wcdawn/container/slepc

#### make config \
####   MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES CC=mpixlc CXX=mpixlC \
####   MPICC=mpixlc MPICXX=mpixlC \
####   HYPRE_DIR=/ccs/proj/csc404/wcdawn/container/hypre-2.22.1/src/hypre \
####   METIS_DIR=/ccs/proj/csc404/wcdawn/container/metis-5.1.0 MFEM_USE_METIS_5=YES \
####   MFEM_USE_CUDA=YES CUDA_CXX=nvcc \
####   CUDA_FLAGS='-x=cu -arch=sm_70 --expt-extended-lambda --compiler-options -O3' \
####   CUDA_LIB='-lcusparse -lcurand -lcublas'
