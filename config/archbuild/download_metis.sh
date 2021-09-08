#!/bin/sh

# requires curl

METIS_VERSION=5.1.0
METIS_TARBALL=metis-${METIS_VERSION}.tar.gz
curl glaros.dtc.umn.edu/gkhome/fetch/sw/metis/${METIS_TARBALL} > ${METIS_TARBALL}
