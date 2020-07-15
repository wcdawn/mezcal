#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"

Result diffusion_hypre_lobpcg(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace);

Result diffusion_slepc(Timer& time, const Input& inp,
                       const Material_library& mat_lib,
                       const mfem::Array<int>& ess_bdr,
                       const mfem::ParMesh* const pmesh,
                       mfem::ParFiniteElementSpace* const fespace);

Result diffusion_power_iteration(Timer& time, const Input& inp,
                                 const Material_library& mat_lib,
                                 const mfem::Array<int>& ess_bdr,
                                 const mfem::ParMesh* const pmesh,
                                 mfem::ParFiniteElementSpace* const fespace);

Result diffusion_fixed_src(Timer& time, const Input& inp,
                           const Material_library& mat_lib,
                           const mfem::Array<int>& ess_bdr,
                           const mfem::ParMesh* const pmesh,
                           mfem::ParFiniteElementSpace* const fespace);

// note: matrix assembly does not include rebuilding RHS in power iteration
// method
const std::string timing_matrix_assembly{"Matrix assembly"};
const std::string timing_solver{"Solver"};

#endif
