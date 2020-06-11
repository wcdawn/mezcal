#!/bin/bash

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
MFEM_BUILD=$WD'/build/mfem'
HYPRE_DIR=$WD'/hypre'
METIS_DIR=$WD'/metis-5.1.0'

git submodule update --init --recursive
git submodule foreach --recursive 'git reset --hard'
git submodule foreach --recursive 'git pull origin master'

echo "BUILD METIS"
# Make sure metis is built. This should already be build and unchanged.
cd $METIS_DIR
if ! [ "$(ls -A ./build)" ] # if the build director is empty
then
  make config CC=${MPICC} CXX=${MPICXX}
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
  ./configure --disable-fortran CXX=${MPICXX} CC=${MPICC}
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

  MFEM_MPICXX="${MPICXX}"

  NVCC=nvcc
  CUDA=''
  if [ -x "$(command -v $NVCC)" ]
  then
    CUDA='MFEM_USE_CUDA=YES'
    MFEM_MPICXX="${MPICXX}"
  fi

  make config CXX=${CXX} MPICXX=${MFEM_MPICXX} CUDA_CXX="nvcc" \
    BUILD_DIR=${MFEM_BUILD} MFEM_USE_MPI=YES \
    MFEM_USE_METIS_5=YES METIS_DIR=@MFEM_DIR@/../metis-5.1.0 ${CUDA}

fi

echo "BUILD MFEM"
cd $MFEM_BUILD
make ${MAKE_FLAGS}

cd $WD
