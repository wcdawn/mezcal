#!/bin/bash

# rely on environment variables:
# - PETSC_DIR
# - PETSC_ARCH
# - Appropriate LD_LIBRARY_PATH (cuda & PETSc)

set -x

CUDA_DIR=/opt/cuda

CXX=g++
CC=gcc

MPICXX=/opt/mpich/bin/mpicxx
MPICC=/opt/mpich/bin/mpicc

usage ()
{
  echo 'Usage:'
  echo "$0 [-jNPROC]"
  echo '[-jNPROC] Number of processors for make compiling.'
}

check_dependency ()
{
  if ! [ -x "$(command -v $1)" ]
  then
    echo "it appears the command '$1' is not on the system"
    echo 'this is a dependency for this script'
    exit 1
  fi
}

DEPENDENCIES=('git' 'make' "${MPICXX}" "${MPICC}" "${CC}" "${CXX}")
for COMMAND in ${DEPENDENCIES[@]}
do
  check_dependency $COMMAND
done

MAKE_FLAGS=''
while getopts "j:h" opt # trailing colon specifies an argument is required
do
  case ${opt} in
    j) # parallel make
      MAKE_FLAGS="-j${OPTARG}"
      ;;
    h) # help
      usage
      exit 0
      ;;
    /?) # error
      echo "Invalid command line argument."
      usage
      exit 1
      ;;
  esac
done

# this isn't really necessary but may be important in the future
shift $((OPTIND - 1)) # shift to ignore already parsed command line arguments
  
WD=$(pwd)
MFEM_DIR=$WD'/mfem'
HYPRE_DIR=$WD'/hypre'
METIS_DIR=$WD'/metis'

git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'
git submodule foreach --recursive 'git pull origin master'

echo "BUILD METIS"
# Make sure metis is built. This should already be build and unchanged.
cd $METIS_DIR
if ! [ "$(ls -A ./build)" ] # if the build director is empty
then
  make config CC="${MPICC} -cc=${CC}" CXX="${MPICXX} -cxx=${CXX}"
  make ${MAKE_FLAGS} || { echo "Error building METIS."; exit 1; }
  # When using metis-5.1.0, we need to create a special link.
  mkdir -p lib && cd lib
  ln -sf ../build/*/libmetis/libmetis.a lib
fi

echo "BUILD HYPRE"
# Rebuild hypre as necessary.
cd $HYPRE_DIR/src

if [ ! -f "./config/Makefile.config" ]
then
  ./configure --disable-fortran CC="${MPICC} -cc=${CC}" CXX="${MPICXX} -cxx=${CXX}"
fi

# Note: the Makefiles for hypre do a lot of copying even if all the targets
# are already built. This messes with MFEM because the timestamp on all the
# HYPRE headers is changed. Thus, the copy command is exchanged for one that
# preserves timestamps.
sed -i 's/cp -fR/cp -pfR/g' ./*/Makefile ./Makefile ./distributed_ls/pilut/Makefile

make ${MAKE_FLAGS} || { echo "Error building HYPRE."; exit 1; }

# Add gitignore file to directory to cleanup some.
ln -sf $HYPRE_DIR/../.gitignore_hypre $HYPRE_DIR/.gitignore

# Rebuild MFEM as necessary.
# configure MFEM if necessary
if [ ! -f "$MFEM_BUILD/GNUmakefile" ]
then

  cd $MFEM_DIR

  MFEM_MPICXX="${MPICXX} -cxx=${CXX}"

  NVCC=nvcc
  CUDA='NO'
  CUDA_FLAGS_VAR=''
  if [ -x "$(command -v $NVCC)" ]
  then
    CUDA='YES'
    CUDA_FLAGS_VAR="-x=cu -arch=sm_60 --expt-extended-lambda --compiler-options -Wall --compiler-options -O3 --compiler-options -cxx=${CXX}"
    MFEM_MPICXX="${MPICXX}"
  fi

  make config CXX=${CXX} MPICXX=${MFEM_MPICXX} CUDA_CXX='nvcc' \
    MFEM_USE_MPI=YES MFEM_USE_OPENMP=YES \
    MFEM_USE_METIS_5=YES METIS_DIR='@MFEM_DIR@/../metis' \
    HYPRE_DIR='@MFEM_DIR@/../hypre/src/hypre' \
    MFEM_USE_CUDA=YES CUDA_FLAGS='-x=cu -arch=sm_60 --expt-extended-lambda' \
    CUDA_LIB='-L/opt/cuda/lib64 -lcusparse -lcusolver -lcublas -lnvToolsExt' \
    MFEM_USE_AMGX=YES AMGX_OPT='-I/home/wcdawn/projects/amgx/base/include' \
    AMGX_LIB='-L/home/wcdawn/projects/amgx/build -lamgx' \
    MFEM_USE_PETSC=YES PETSC_DIR=${PETSC_DIR}/${PETSC_ARCH} PETSC_ARCH=${PETSC_ARCH} \
    MFEM_USE_SLEPC=YES SLEPC_DIR=/home/wcdawn/projects/slepc

  # configure using SLEPc
  # PETSC_ARCH is necessary for SLEPc because MFEM has a weird default value 
  # ("arch-linux2-c-debug")
####  make config MPICXX=/opt/mpich/bin/mpicxx BUILD_DIR=../build/mfem \
####    MFEM_USE_MPI=YES MFEM_USE_METIS_5=YES METIS_DIR='@MFEM_DIR@/../metis' \
####    MFEM_USE_CUDA=YES \
####    MFEM_USE_PETSC=YES PETSC_DIR=/home/wcdawn/projects/petsc/arch-linux-c-debug/ \
####    MFEM_USE_SLEPC=YES SLEPC_DIR=/home/wcdawn/projects/slepc PETSC_ARCH=arch-linux-c-debug

fi

echo "BUILD MFEM"
cd $MFEM_BUILD
make ${MAKE_FLAGS}

cd $WD
