#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "xslibrary.hpp"
#include "material_library.hpp"

Result diffusion_lobpcg(const Input& inp, const Material_library& mat_lib,
                        const mfem::Array<int>& ess_bdr,
                        const mfem::ParMesh* const pmesh,
                        mfem::ParFiniteElementSpace* const fespace);

Result diffusion_power_iteration(const Input& inp,
                                 const Material_library& mat_lib,
                                 const mfem::Array<int>& ess_bdr,
                                 const mfem::ParMesh* const pmesh,
                                 mfem::ParFiniteElementSpace* const fespace);

Result diffusion_fixed_src(const Input& inp, const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           const mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace);

#endif
