#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "mfem.hpp"
#include "xslibrary.hpp"

struct Result
{
  Result(mfem::ParFiniteElementSpace* fespace) 
  { 
    phi.SetSpace(fespace); 
    phi = 0.0;
  }
  mfem::ParGridFunction phi;
  double keff{1.0};
};

Result diffusion_lobpcg(const Input& inp, const XSLibrary& xs,
                        const mfem::Array<int>& ess_bdr,
                        const mfem::ParMesh* const pmesh,
                        mfem::ParFiniteElementSpace* const fespace);

Result diffusion_power_iteration(const Input& inp, const XSLibrary& xs,
                                 const mfem::Array<int>& ess_bdr,
                                 const mfem::ParMesh* const pmesh,
                                 mfem::ParFiniteElementSpace* const fespace);

Result diffusion_fixed_src(const Input& inp, const XSLibrary& xs,
                           const mfem::Array<int>& ess_bdr,
                           const mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace);

#endif
