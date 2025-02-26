#!/bin/sh

METIS_VERSION=5.1.0

# this is the METIS version maintained by the MFEM developers
# they recommend using this instead of the original one, mostly for stability of downloads
# no really good reason for using v5, just seems fine...

curl -v -L https://github.com/mfem/tpls/raw/gh-pages/metis-${METIS_VERSION}.tar.gz > ../metis_v${METIS_VERSION}.tar.gz
cd ..
gunzip metis_v${METIS_VERSION}.tar.gz
tar -xvf metis_v${METIS_VERSION}.tar
rm -f metis_v${METIS_VERSION}.tar
