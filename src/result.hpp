#ifndef RESULT
#define RESULT

#include "mfem.hpp"

struct Result
{
  Result(mfem::ParFiniteElementSpace* fespace) 
  { 
    phi.SetSpace(fespace); 

    // the following block sets phi=1 and phi=0 at essential bcs
    phi = 0.0;
    phi.SetTrueVector();
    phi = 1.0;
    phi.SetFromTrueVector();
  }
  mfem::ParGridFunction phi;
  double keff{1.0};
};

#endif
