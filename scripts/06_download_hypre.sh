#!/bin/sh

HYPRE_VERSION=2.26.0 # October 2022

curl -v -L https://github.com/hypre-space/hypre/archive/refs/tags/v${HYPRE_VERSION}.tar.gz > ../hypre_v${HYPRE_VERSION}.tar.gz
cd ..
gunzip hypre_v${HYPRE_VERSION}.tar.gz
tar -xvf hypre_v${HYPRE_VERSION}.tar
rm -f hypre_v${HYPRE_VERSION}.tar
