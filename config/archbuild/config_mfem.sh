#!/bin/sh

#### make config \
####   MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES CC=mpicc CXX=mpicxx \
####   MPICC=mpicc MPICXX=mpicxx \
####   HYPRE_DIR=/home/wcdawn/projects/container/hypre-2.22.0/src/hypre \
####   METIS_DIR=/home/wcdawn/projects/container/metis-5.1.0 MFEM_USE_METIS_5=YES \
####   MFEM_USE_CUDA=YES CUDA_CXX=nvcc \
####   CUDA_FLAGS='-x=cu -arch=sm_60 --expt-extended-lambda --compiler-options -O3' \
####   CUDA_LIB='-lcusparse -lcurand -lcublas -lnvToolsExt' \
####   MFEM_USE_PETSC=YES \
####   PETSC_DIR=/home/wcdawn/projects/container/petsc/arch-linux-c-opt \
####   PETSC_ARCH=arch-linux-c-opt \
####   MFEM_USE_SLEPC=YES \
####   SLEPC_DIR=/home/wcdawn/projects/container/slepc \
####   MFEM_USE_AMGX=YES \
####   AMGX_OPT='-I/home/wcdawn/projects/amgx/base/include' \
####   AMGX_LIB='-L/home/wcdawn/projects/amgx/build -lamgx'

make config \
  MFEM_USE_OPENMP=YES MFEM_USE_MPI=YES CC=mpicc CXX=mpicxx \
  MPICC=mpicc MPICXX=mpicxx \
  HYPRE_DIR=/home/wcdawn/projects/container/hypre-2.22.1/src/hypre \
  METIS_DIR=/home/wcdawn/projects/container/metis-5.1.0 MFEM_USE_METIS_5=YES \
  MFEM_USE_CUDA=YES CUDA_CXX=nvcc \
  CUDA_FLAGS='-x=cu -arch=sm_75 --expt-extended-lambda --compiler-options -O3' \
  CUDA_LIB='-lcusparse -lcurand -lcublas -lcusolver -lnvToolsExt' \
  MFEM_USE_PETSC=YES \
  PETSC_DIR=/home/wcdawn/projects/container/petsc/arch-linux-c-opt \
  PETSC_ARCH=arch-linux-c-opt \
  MFEM_USE_SLEPC=YES \
  SLEPC_DIR=/home/wcdawn/projects/container/slepc \
  MFEM_USE_AMGX=YES \
  AMGX_OPT='-I/home/wcdawn/projects/amgx/base/include' \
  AMGX_LIB='-L/home/wcdawn/projects/amgx/build -lamgx'
