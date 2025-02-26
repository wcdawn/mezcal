#!/bin/sh

SLEPC_VERSION=3.18.1
SLEPC_VERSION=3.20.0

curl -v -L https://slepc.upv.es/download/distrib/slepc-${SLEPC_VERSION}.tar.gz > ../slepc_v${SLEPC_VERSION}.tar.gz
cd ..
gunzip slepc_v${SLEPC_VERSION}.tar.gz
tar -xvf slepc_v${SLEPC_VERSION}.tar
rm -f slepc_v${SLEPC_VERSION}.tar
