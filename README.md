
```

  __  __ ______ ___________          _              ;"--,   / \   ,--";
 |  \/  |  ____|___  / ____|   /\   | |       ,'"--,;    ' /   \ '    ;,--"',
 | \  / | |__     / / |       /  \  | |       ",    '"--, /     \ ,--"'    ,"
 | |\/| |  __|   / /| |      / /\ \ | |        '"-,      |       |      ,-"'
 | |  | | |____ / /_| |____ / ____ \| |____        '"--, \       / ,--"'
 |_|  |_|______/_____\_____/_/    \_\______|      __..:::::::::::::::..__
         Microreactor eZ CALculation

```

# Description

This will be the program to model microreactors for my Ph.D. dissertation.
This project is in association with Oak Ridge National Laboratory (ORNL) and the [Exascale Computing Project](https://exascaleproject.org).
This work is based on the [MFEM](https://mfem.org) finite element framework developed at Lawrence Livermore National Laboratory (LLNL).

# Building

## Dependencies

For C/C++/Fortran compilers, use GCC compiler suite. Tested with v6.4, v9.1, and v11.1.
For MPI implementation, use any of OpenMPI, MPICH, or Spectrum MPI (e.g., `mpixlc`).
Compiling has been tested with both CUDA v10 and v11.
The following dependencies are listed in the order in which they should be built.

1. [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview).

    I always use METIS v5.1 but results should be similar with v4.

1. [HYPRE](https://github.com/hypre-space/hypre).

    Especially when using GPU options, it is important to use the most up-to-date version possible.
    Typically, using `hypre:master` is good unless something breaks and an older commit can be used instead. At the time of writing this, v2.22.1 is the newest HYPRE release and works well.
    HYPRE should be compiled with CUDA options. In the near future, it will be necessary to compile with at least `MixedInt` if not `BigInt` as well.

1. [PETSc](https://gitlab.com/petsc/petsc).

    Using `petsc:main` is typically fine. None of the PETSc options used are particularly new.
    PETSc uses shared libraries by default so it is important to set `LD_LIBRARY_PATH` appropriately when running executables.

1. [SLEPc](https://gitlab.com/slepc/slepc).

    Using `slepc:main` is typically fine. None of the SLEPc options used are particularly new.
    SLEPc uses shared libraries by default so it is important to set `LD_LIBRARY_PATH` appropriately when running executables.

1. (optional) [AmgX](https://github.com/NVIDIA/AMGX).

    AmgX should be an optional dependence and code depednent on AmgX should only be compiled if `AMGX_AVAILABLE` is defined (e.g., `-DAMGX_AVAILABLE`). I have not thoroughly tested compiling without AmgX. I have tested with v 2.0, v2.1, and v2.2.
    
1. [MFEM](https://github.com/mfem/mfem).

    MFEM should be compiled with OpenMP and all of the above dependencies.

## Configuration

### Dependencies

Example configuration files are provided in the `/config` directory for Summit and my PC.

### `MEZCAL`

MEZCAL doesn't have many configuration options. Only `-DGPU_AWARE_MPI` is which useful on Summit and `-DAMGX_AVAILABLE` can be used to specify whether the AmgX library is available in an environment.

All of the configuration in MEZCAL is intended to be performed using the `/Makefile.include` file. The Makefile uses an environmental variable `MACH` to configure for a certain system. This is typically something specified in a users `$HOME/.bashrc` file. For example
```sh
# $HOME/.bashrc contents:

export MACH='SUMMIT'
```
The `/Makefile.include` file can be edited to point to the appropriate libraries in a particular environment.

# Source Files

The source code is in the `/src` directory with the exception of some vector versions of `mfem::BilinearFormIntegrator` and `mfem::LinearFormIntegrator` that I needed to write which are included in the `/addon` directory.
The code itself is typically written in Fortran/C style using `void` functions and output-parameters.
Eventually, it may be nice to use a more Object Oriented (OO) approach; but, I don't have time to prototype algorithms and worry about move semantics and Return Value Optimization (RVO) in C++ at the same time.

Hopefully, most of the filenames should be self explanatory.
Here are a few of the important files:
- `/src/solver_saaf_angle_power_iteration.[ch]pp` Source Iteration solver and associated functions.
- `/src/solver_saaf_true_slepc_vector.[ch]pp` Generalized Eigensolver and associated functions.
- `/src/solver.[ch]pp` other solver algorithms I have investigated along the way. This also contains some duplicate code.

# Acknowledgments

This research was supported by the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration.

This research used resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory, which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725.
