#ifndef RESULT
#define RESULT

#include "mfem.hpp"

struct Result
{
  Result(mfem::ParFiniteElementSpace* fespace, const mfem::Array<int>& ess_tdof_list)
  { 
    phi.SetSpace(fespace); 

    // the following block sets phi=1 and phi=0 at essential bcs
    phi = 1.0;
    for (int i{0}; i < ess_tdof_list.Size(); ++i)
      phi[ess_tdof_list[i]] = 0.0;
  }
  mfem::ParGridFunction phi;
  double keff{1.0};
};

#endif
