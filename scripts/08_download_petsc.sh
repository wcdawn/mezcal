#!/bin/sh

PETSC_VERSION=3.18.1
PETSC_VERSION=3.20.0

#curl -v -L https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-${PETSC_VERSION}.tar.gz > ../petsc_v${PETSC_VERSION}.tar.gz
curl -v -L https://web.cels.anl.gov/projects/petsc/download/release-snapshots/petsc-${PETSC_VERSION}.tar.gz > ../petsc_v${PETSC_VERSION}.tar.gz
 

cd ..
gunzip petsc_v${PETSC_VERSION}.tar.gz
tar -xvf petsc_v${PETSC_VERSION}.tar
rm -f petsc_v${PETSC_VERSION}.tar
