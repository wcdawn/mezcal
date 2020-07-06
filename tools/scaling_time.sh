#!/bin/sh

NPROC=24
OUTNAME='./times.txt'

MPIRUN='/opt/mpich/bin/mpirun'
EXEC="../../src/micro.x"
INPUT='bwr.inp'
TIME='/usr/bin/time'

rm -f $OUTNAME

for N in $(seq $NPROC)
do
  command time -f '%e' -o $OUTNAME -a $MPIRUN -np $N $EXEC -i $INPUT -rs 5 || exit 1
done
