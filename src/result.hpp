#ifndef RESULT
#define RESULT

#include "mfem.hpp"

struct Result
{
  Result(mfem::ParFiniteElementSpace* fespace)
  { 
    phi.SetSpace(fespace); 
    // must be zero to enforce zero-flux (essential) boundary conditions
    phi = 0.0;
  }
  mfem::ParGridFunction phi;
  double keff{1.0};
};

#endif
