#!/bin/bash

EXAMPLE_NAME=ex11p

nvcc  -O3 -std=c++11 -std=c++11 -x=cu --expt-extended-lambda -arch=sm_60 \
  --compiler-options -Wall \
  --compiler-options -O3 \
  --compiler-options -std=c++11 \
  --compiler-options -cxx=g++ \
  -ccbin /opt/mpich/bin/mpicxx \
  -I.. -I/home/wcdawn/work/microreactor/mfem/../hypre/src/hypre/include \
  -I/home/wcdawn/work/microreactor/mfem/../metis/include \
  -I/home/wcdawn/projects/petsc/include \
  -I/home/wcdawn/projects/petsc/arch-linux-c-debug/include \
  -I/opt/cuda/include \
  -I/opt/mpich/include \
  -I/home/wcdawn/projects/slepc/include \
  -I/home/wcdawn/projects/slepc/arch-linux-c-debug/include \
  /home/wcdawn/work/microreactor/mfem/examples/${EXAMPLE_NAME}.cpp -o ${EXAMPLE_NAME} \
  -L.. -lmfem \
  -L/home/wcdawn/work/microreactor/mfem/../hypre/src/hypre/lib -lHYPRE \
  -L/home/wcdawn/work/microreactor/mfem/../metis/lib -lmetis \
  --compiler-options -Xlinker --compiler-options -rpath --compiler-options -Xlinker --compiler-options /home/wcdawn/projects/petsc/arch-linux-c-debug/lib \
  -L/home/wcdawn/projects/petsc/arch-linux-c-debug/lib -lpetsc \
  --compiler-options -Xlinker --compiler-options -rpath --compiler-options -Xlinker --compiler-options /opt/cuda/lib \
  -L/opt/cuda/lib -llapack -lblas \
  -lpthread -lX11 -lm -lcufft -lcublas -lcudart -lcusparse -lcusolver -lcuda -lquadmath -lstdc++ -ldl \
  --compiler-options -Xlinker --compiler-options -rpath --compiler-options -Xlinker --compiler-options /home/wcdawn/projects/slepc/arch-linux-c-debug/lib \
  -L/home/wcdawn/projects/slepc/arch-linux-c-debug/lib -lslepc        -lrt 
