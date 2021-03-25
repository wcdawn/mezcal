#ifndef SOLVER_AMGX_H
#define SOLVER_AMGX_H

#include "input.hpp"
#include "material_library.hpp"

#include "mfem.hpp"
#include "../addon/mfem_addon.hpp"


void spn_amgx_cuda_vector(Timer& time, const Input& inp,
                     Material_library& mat_lib,
                     const mfem::Array<int>& ess_bdr,
                     const mfem::ParMesh* const pmesh,
                     mfem::ParFiniteElementSpace* const fespace,
                     Result& res);


#endif
