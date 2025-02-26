#!/bin/sh

echo "r0"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r0.mesh | grep 'keff = '
cp out_000000.csv ./out_r0.csv

echo "r1"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r1.mesh | grep 'keff = '
cp out_000000.csv ./out_r1.csv

echo "r2"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r2.mesh | grep 'keff = '
cp out_000000.csv ./out_r2.csv

echo "r3"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r3.mesh | grep 'keff = '
cp out_000000.csv ./out_r3.csv

echo "r4"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r4.mesh | grep 'keff = '
cp out_000000.csv ./out_r4.csv

echo "r5"
../src/mezcal.x -i ./twomat.inp --mesh ./oneD_r5.mesh | grep 'keff = '
cp out_000000.csv ./out_r5.csv
