#!/bin/sh

MFEM_VERSION=4.5

curl -v -L https://github.com/mfem/mfem/archive/refs/tags/v${MFEM_VERSION}.tar.gz > ../mfem_v${MFEM_VERSION}.tar.gz
cd ..
gunzip mfem_v${MFEM_VERSION}.tar.gz
tar -xvf mfem_v${MFEM_VERSION}.tar
rm -f mfem_v${MFEM_VERSION}.tar
