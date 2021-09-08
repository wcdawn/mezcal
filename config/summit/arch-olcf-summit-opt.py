#!/usr/bin/python3

# Example configure script for the IBM POWER9 and NVIDIA Volta GV100 "Summit" system at OLCF/ORNL.
# This may also be useful for the related Sierra system at LLNL, or other, similar systems that may appear.
# A compiler module and the 'cmake' and 'cuda' modules should be loaded on Summit.
# See inline comments below on other modules that might need to be loaded. 

if __name__ == '__main__':
  import os
  import sys
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    # We use the IBM Spectrum MPI compiler wrappers, regardless of the underlying compilers used.
    '--with-cc=mpicc',
    '--with-cxx=mpiCC',
    '--with-fc=mpifort',
    '--with-make-np=32',

    '--with-shared-libraries=1',

    '--with-debugging=no',

    '--COPTFLAGS=-g -Ofast -mcpu=power9',
    '--CXXOPTFLAGS=-g -Ofast -mcpu=power9',
    '--FOPTFLAGS=-g -Ofast -mcpu=power9',

    '--download-fblaslapack=1',

    #'--download-hypre=1',
    # TODO figure out hypre

  ] 
  configure.petsc_configure(configure_options)
