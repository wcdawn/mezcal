The general directory structure should be

```
CONATAINER
    |
    | - environment.sh
    | - scripts
    | - hdf5-1.12.1
    | - hypre-2.26.0
    | - metis-5.1.0
    | - mezcal
    \
      | - src/mezcal.x
    | - mfem
    | - mfem-4.5
    | - mpich-4.0.3
    | - mpich-install
    | - petsc-3.20.0
    | - scripts
    | - slepc-3.20.0
```

You need the scripts to build everything.
So I would do something like
```
$ cd $CONTAINER && git clone git@github.com:wcdawn/mezcal && cp -r ./mezcal/scripts ./ && rm -rvf ./mezcal
```
So that you can clone it later. Dumb, I know.

Run them in order

1. `00_download_mpich.sh`
1. `01_install_mpich.sh`
1. `02_download_hdf5.sh`
1. `03_install_hdf5.sh`
1. `04_download_metis.sh`
1. `05_installl_metis.sh`
1. `06_download_hypre.sh`
1. `07_install_hypre.sh`
1. `08_download_petsc.sh`
1. `09_install_petsc.sh`
1. `10_download_slepc.sh`
1. `11_install_slepc.sh`
1. `12_download_mfem.sh`
1. `13_install_mfem.sh`
1. `14_download_mezcal.sh`

You'll need ssh permission to the MEZCAL repository to use the `11_download_mezcal.sh` script.
You'll probably also need to set the `MEZCAL_MACH` environmental variable to a new value and edit `Makefile.include` to point to the right places.
Finally, `cd ../mezcal/src && make`.

You need to set the environmental variables to contain the PETSC and SLEPC paths since these are dynamically linked libraries.
I won't know the path ahead of time, but you can use `make_environment.sh` to write a shell script `../enviroment.sh` that you should run/source before you want to run MEZCAL.

The final run-line for running the "twomat" problem
```
$ cd ./mezcal/demo/twomat
$ ../../../mpich-install/bin/mpirun -n 3 ../../src/mezcal.x -i ./twomat.inp
```
You should see something like `keff = 0.99390479860911451482`.
