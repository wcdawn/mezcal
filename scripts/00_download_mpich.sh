#!/bin/sh

MPICH_VERSION=4.0.3

curl -v -L https://www.mpich.org/static/downloads/${MPICH_VERSION}/mpich-${MPICH_VERSION}.tar.gz > ../mpich_v${MPICH_VERSION}.tar.gz
cd ..
gunzip mpich_v${MPICH_VERSION}.tar.gz
tar -xvf mpich_v${MPICH_VERSION}.tar
rm -f mpich_v${MPICH_VERSION}.tar
