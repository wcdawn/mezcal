#!/bin/sh

HDF5_VERSION=1.12.1

curl -v -L https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.1/src/hdf5-1.12.1.tar.gz > ../hdf5-1.12.1.tar.gz
cd ..
gunzip hdf5-1.12.1.tar.gz
tar -xvf hdf5-1.12.1.tar
rm -f hdf5-1.12.1.tar
