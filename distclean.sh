#!/bin/bash

WD=$(pwd)
MFEM_DIR=$WD'/mfem'
MFEM_BUILD=$WD'/build/mfem'
HYPRE_DIR=$WD'/hypre'
METIS_DIR=$WD'/metis-5.1.0'

cd ${MFEM_DIR}
make distclean

cd ${MFEM_BUILD}
rm GNUmakefile

cd ${HYPRE_DIR}/src
make distclean

cd ${METIS_DIR}
make distclean
