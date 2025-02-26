#!/bin/sh

PWD=$(pwd)
PETSC_VERSION=3.20.0
PETSC_DIR=${PWD}/../petsc-${PETSC_VERSION}
PETSC_ARCH=arch-linux-c-debug
SLEPC_DIR=${PWD}/../slepc-${PETSC_VERSION}

echo '#!/bin/sh' > ../environment.sh
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'"${PWD}/../petsc-${PETSC_VERSION}/${PETSC_ARCH}/lib/" >> ../environment.sh
echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:'"${PWD}/../slepc-${PETSC_VERSION}/${PETSC_ARCH}/lib/" >> ../environment.sh
