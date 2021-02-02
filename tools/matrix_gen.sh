#!/bin/bash

EXEC=../../src/micro.x
INPUT=bwr.inp

for REFINE in {0..4}
do
  $EXEC -i $INPUT -rs $REFINE
  cp ./petsca_vector ./a_matrix_r$REFINE
  cp ./petscm_vector ./b_matrix_r$REFINE
done
