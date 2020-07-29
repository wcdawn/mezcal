#ifndef SOLVER
#define SOLVER

#include "input.hpp"
#include "material_library.hpp"
#include "mfem.hpp"
#include "result.hpp"
#include "timer.hpp"
#include "xslibrary.hpp"

#include <exception>
#include <string>

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

Result diffusion_slepc_vector(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace);

Result diffusion_slepc_manual(Timer& time, const Input& inp,
                              const Material_library& mat_lib,
                              const mfem::Array<int>& ess_bdr,
                              const mfem::ParMesh* const pmesh,
                              mfem::ParFiniteElementSpace* const fespace);

class Failed_to_converge : std::exception
{
  public:
    Failed_to_converge () {}
    Failed_to_converge (const std::string& message) : 
      msg{"Failed to converge.\n" + message} {}

    const char* what() const noexcept override
    {
      return msg.c_str();
    }

  private:
    const std::string msg;
};

#endif
