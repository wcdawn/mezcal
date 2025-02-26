#!/bin/sh

PWD=$(pwd)
CC=${PWD}/../mpich-install/bin/mpicc
CXX=${PWD}/../mpich-install/bin/mpicxx

METIS_VERSION=5.1.0
HYPRE_VERSION=2.26.0
MFEM_VERSION=4.5
PETSC_VERSION=3.18.0
PETSC_VERSION=3.20.0

PETSC_DIR=${PWD}/../petsc-${PETSC_VERSION}
PETSC_ARCH=arch-linux-c-debug
SLEPC_DIR=${PWD}/../slepc-${PETSC_VERSION}

NPROC='10' # need to limit memory usage

cd ../mfem-${MFEM_VERSION}
make config \
  CC=${CC} CXX=${CXX} \
  MFEM_USE_MPI=YES MPICXX=${CXX} \
  MFEM_USE_METIS_5=YES METIS_DIR=${PWD}/../metis-${METIS_VERSION} \
  HYPRE_DIR=${PWD}/../hypre-${HYPRE_VERSION}/src/hypre \
  MFEM_USE_PETSC=YES \
  PETSC_DIR=${PETSC_DIR}/${PETSC_ARCH} PETSC_ARCH=${PETSC_ARCH} \
  PETSC_LIB='-L/home/wcdawn/projects/container_cpu/petsc-3.20.0/arch-linux-c-debug/lib -Wl,-rpath,/home/wcdawn/projects/container_cpu/hypre-2.26.0/src/hypre/lib -L/home/wcdawn/projects/container_cpu/hypre-2.26.0/src/hypre/lib -L/usr/lib/gcc/x86_64-pc-linux-gnu/13.2.1 -lpetsc -lHYPRE -llapack -lblas -lm -lX11 -lmpi_usempif08 -lmpi_usempi_ignore_tkr -lmpi_mpifh -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++ -lquadmath ' \
  MFEM_USE_SLEPC=YES SLEPC_DIR=${SLEPC_DIR} || exit 1
make config \
  CC=${CC} CXX=${CXX} \
  MFEM_USE_MPI=YES MPICXX=${CXX} \
  MFEM_USE_METIS_5=YES METIS_DIR=${PWD}/../metis-${METIS_VERSION} \
  HYPRE_DIR=${PWD}/../hypre-${HYPRE_VERSION}/src/hypre \
  MFEM_USE_PETSC=YES \
  PETSC_DIR=${PETSC_DIR}/${PETSC_ARCH} PETSC_ARCH=${PETSC_ARCH} \
  MFEM_USE_SLEPC=YES SLEPC_DIR=${SLEPC_DIR} || exit 1

make -j${NPROC}
cd ..
ln -sf ./mfem-${MFEM_VERSION} ./mfem
